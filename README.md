# Project Overview

This project is based on the book "Programming Massively Parallel Processors: A Hands-on Approach" by Wen-mei W. Hwu, David B. Kirk, and Izzat El Hajj. The files in the `/code` directory broadly correspond to daily progress through the book's material. The code in this project was generated with Google AI Studio, Gemini CLI, and Qwen Code.

---

# CUDA Programming Experiments

This project contains CUDA programming experiments using the leetgpu cloud service, which allows CUDA development and testing on systems without NVIDIA GPU hardware (like Mac laptops).

## Project Structure
- `/code`: CUDA source files for experiments. Each subdirectory contains a `SUMMARY.md` file that provides a brief overview of the concepts covered in that section.

## Mental Models
The `mmx.md` files in each of the `code` subdirectories contain my mental models of the corresponding CUDA code. These are my notes and understanding of the concepts as I progress through the material.

## Getting Started
1. Write your CUDA code in a `.cu` file and place it in the `/code` directory
2. Use leetgpu to run and test your code:
   ```bash
   leetgpu run code/your_file.cu  # Runs in functional mode by default
   ```
3. When I say to run a specific file, I will execute the leetgpu command with that file so you can see the output and any errors that occur

## About leetgpu
leetgpu is a cloud-based GPU service that we use via its playground and CLI to test and iterate on the code. It provides:
- Functional mode for fast correctness testing
- Cycle-accurate mode for performance analysis
- Access to multiple GPU models (NVIDIA GV100, TITAN V, RTX series, etc.)
- Support for major CUDA features while having some limitations
