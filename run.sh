#!/bin/bash
mkdir -p data output logs

echo "Building..."
make build

echo "Running batch image processor..."
./image_processor.exe -i data -o output -b 2 | tee logs/execution.log

echo "Done. Check output/ for results and logs/execution.log for details."
