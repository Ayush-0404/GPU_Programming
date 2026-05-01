NVCC = nvcc
NVCC_FLAGS = --std c++17 -I./src
SRC = src/main.cu
TARGET = image_processor.exe

.PHONY: clean build run all

build:
	$(NVCC) $(NVCC_FLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
	rm -rf output/*

run:
	./$(TARGET) -i data -o output -b 2 | tee logs/execution.log

all: clean build run
