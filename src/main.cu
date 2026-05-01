// Copyright 2024 CUDA Batch Image Processor
// Converts RGB images to grayscale using GPU kernels

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// CUDA kernel: RGB to Grayscale using luminosity method
__global__ void RgbToGrayscaleKernel(const unsigned char* input,
                                      unsigned char* output,
                                      int width, int height,
                                      int channels) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int total_pixels = width * height;

  if (idx < total_pixels) {
    int rgb_idx = idx * channels;
    unsigned char r = input[rgb_idx];
    unsigned char g = input[rgb_idx + 1];
    unsigned char b = input[rgb_idx + 2];
    // ITU-R BT.601 luminosity weights
    output[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
  }
}

// CUDA kernel: Apply simple box blur on grayscale image
__global__ void BoxBlurKernel(const unsigned char* input,
                               unsigned char* output,
                               int width, int height,
                               int radius) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x < width && y < height) {
    int sum = 0;
    int count = 0;
    for (int dy = -radius; dy <= radius; dy++) {
      for (int dx = -radius; dx <= radius; dx++) {
        int nx = x + dx;
        int ny = y + dy;
        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
          sum += input[ny * width + nx];
          count++;
        }
      }
    }
    output[y * width + x] = (unsigned char)(sum / count);
  }
}

// Check if filename has image extension
int IsImageFile(const char* filename) {
  const char* ext = strrchr(filename, '.');
  if (!ext) return 0;
  return (strcasecmp(ext, ".jpg") == 0 ||
          strcasecmp(ext, ".jpeg") == 0 ||
          strcasecmp(ext, ".png") == 0 ||
          strcasecmp(ext, ".bmp") == 0);
}

void PrintUsage(const char* prog) {
  printf("Usage: %s -i <input_dir> -o <output_dir> [-b blur_radius]\n", prog);
  printf("  -i  Input directory containing images\n");
  printf("  -o  Output directory for processed images\n");
  printf("  -b  Optional box blur radius (default: 0 = no blur)\n");
}

int main(int argc, char* argv[]) {
  char input_dir[512] = "";
  char output_dir[512] = "";
  int blur_radius = 0;

  // Parse CLI arguments
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
      strncpy(input_dir, argv[++i], 511);
    } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
      strncpy(output_dir, argv[++i], 511);
    } else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) {
      blur_radius = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-h") == 0) {
      PrintUsage(argv[0]);
      return 0;
    }
  }

  if (strlen(input_dir) == 0 || strlen(output_dir) == 0) {
    PrintUsage(argv[0]);
    return 1;
  }

  // Create output directory
  mkdir(output_dir, 0755);

  // Query GPU
  int device_count = 0;
  cudaGetDeviceCount(&device_count);
  printf("Number of CUDA devices: %d\n", device_count);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Using device: %s\n", prop.name);
  cudaSetDevice(0);

  printf("Input directory: %s\n", input_dir);
  printf("Output directory: %s\n", output_dir);
  printf("Blur radius: %d\n", blur_radius);
  printf("-------------------------------------------\n");

  // Scan input directory
  DIR* dir = opendir(input_dir);
  if (!dir) {
    fprintf(stderr, "Cannot open input directory: %s\n", input_dir);
    return 1;
  }

  int total_images = 0;
  int processed = 0;
  int failed = 0;
  double total_gpu_time_ms = 0.0;
  long long total_pixels = 0;

  auto wall_start = std::chrono::high_resolution_clock::now();

  struct dirent* entry;
  while ((entry = readdir(dir)) != NULL) {
    if (!IsImageFile(entry->d_name)) continue;
    total_images++;

    char input_path[1024];
    snprintf(input_path, sizeof(input_path), "%s/%s",
             input_dir, entry->d_name);

    // Load image
    int width, height, channels;
    unsigned char* img = stbi_load(input_path, &width, &height,
                                    &channels, 3);
    if (!img) {
      fprintf(stderr, "Failed to load: %s\n", input_path);
      failed++;
      continue;
    }

    int num_pixels = width * height;
    size_t rgb_size = num_pixels * 3 * sizeof(unsigned char);
    size_t gray_size = num_pixels * sizeof(unsigned char);

    // Allocate device memory
    unsigned char *d_rgb, *d_gray, *d_blurred;
    cudaMalloc(&d_rgb, rgb_size);
    cudaMalloc(&d_gray, gray_size);

    // Copy to device
    cudaMemcpy(d_rgb, img, rgb_size, cudaMemcpyHostToDevice);

    // Time the GPU work
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch grayscale kernel
    int threads = 256;
    int blocks = (num_pixels + threads - 1) / threads;
    RgbToGrayscaleKernel<<<blocks, threads>>>(d_rgb, d_gray,
                                               width, height, 3);

    // Optional blur pass
    unsigned char* d_final = d_gray;
    if (blur_radius > 0) {
      cudaMalloc(&d_blurred, gray_size);
      dim3 block_dim(16, 16);
      dim3 grid_dim((width + 15) / 16, (height + 15) / 16);
      BoxBlurKernel<<<grid_dim, block_dim>>>(d_gray, d_blurred,
                                              width, height,
                                              blur_radius);
      d_final = d_blurred;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    total_gpu_time_ms += gpu_ms;
    total_pixels += num_pixels;

    // Copy result back
    unsigned char* h_gray = (unsigned char*)malloc(gray_size);
    cudaMemcpy(h_gray, d_final, gray_size, cudaMemcpyDeviceToHost);

    // Save output
    char output_path[1024];
    char base_name[512];
    strncpy(base_name, entry->d_name, 511);
    char* dot = strrchr(base_name, '.');
    if (dot) *dot = '\0';

    snprintf(output_path, sizeof(output_path), "%s/%s_gray.png",
             output_dir, base_name);
    stbi_write_png(output_path, width, height, 1, h_gray, width);

    printf("Processed: %s (%dx%d) -> %s [GPU: %.2f ms]\n",
           entry->d_name, width, height, output_path, gpu_ms);

    // Cleanup per image
    stbi_image_free(img);
    free(h_gray);
    cudaFree(d_rgb);
    cudaFree(d_gray);
    if (blur_radius > 0) cudaFree(d_blurred);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    processed++;
  }
  closedir(dir);

  auto wall_end = std::chrono::high_resolution_clock::now();
  double wall_ms = std::chrono::duration<double, std::milli>(
      wall_end - wall_start).count();

  printf("-------------------------------------------\n");
  printf("Total images found: %d\n", total_images);
  printf("Successfully processed: %d\n", processed);
  printf("Failed: %d\n", failed);
  printf("Total pixels processed: %lld\n", total_pixels);
  printf("Total GPU kernel time: %.2f ms\n", total_gpu_time_ms);
  printf("Total wall time: %.2f ms\n", wall_ms);
  if (processed > 0) {
    printf("Average GPU time per image: %.2f ms\n",
           total_gpu_time_ms / processed);
  }
  printf("Done\n");

  cudaDeviceReset();
  return 0;
}
