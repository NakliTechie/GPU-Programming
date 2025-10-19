# Project Overview

This project is based on the book "Programming Massively Parallel Processors: A Hands-on Approach" by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The files in the `/code` directory broadly correspond to daily progress through the book's material.

---

# CUDA Programming Experiments

This project contains CUDA programming experiments using the leetgpu cloud service, which allows CUDA development and testing on systems without NVIDIA GPU hardware (like Mac laptops).

## Project Structure
- `/code`: CUDA source files for experiments
- `/docs`: Documentation, guidelines, and examples for CUDA development with leetgpu

## Getting Started
1. Write your CUDA code in a `.cu` file and place it in the `/code` directory
2. Use leetgpu to run and test your code:
   ```bash
   leetgpu run code/your_file.cu  # Runs in functional mode by default
   ```
3. When I say to run a specific file, I will execute the leetgpu command with that file so you can see the output and any errors that occur
4. See the documentation in `/docs` for more detailed usage instructions

## Documentation
- `docs/README.md`: Complete guidelines for CUDA development with leetgpu
- `docs/quick_reference.md`: Quick reference for common leetgpu commands
- `docs/troubleshooting.md`: Solutions to common issues
- `docs/example_vector_add.cu`: Example CUDA program for testing

## About leetgpu
leetgpu provides:
- Functional mode for fast correctness testing
- Cycle-accurate mode for performance analysis
- Access to multiple GPU models (NVIDIA GV100, TITAN V, RTX series, etc.)
- Support for major CUDA features while having some limitations

For more details about capabilities and usage, see the documentation files in the `/docs` directory.