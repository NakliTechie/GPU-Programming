# Gemini for CUDA Learning

Hello! I'm Gemini, and I'm here to help you with your CUDA learning project using `leetgpu`. I've reviewed the project documentation and I'm ready to assist you.

## My Understanding of the Project

This project is designed for learning and experimenting with CUDA programming in an environment where a physical NVIDIA GPU is not available. You are using `leetgpu`, a cloud-based GPU simulator, to run and test your CUDA code on your Mac.

Here's a quick summary of the key aspects:

*   **`leetgpu` Simulator:** You can run your CUDA code in two modes:
    *   **Functional Mode:** For fast correctness checking (CUDA 11.7).
    *   **Cycle-Accurate Mode:** For detailed performance analysis on various simulated GPUs (CUDA 11.8).
*   **Project Structure:** Your CUDA code is in the `/code` directory, and the documentation is in the `/docs` directory.
*   **Workflow:** On load, I will take stock of all files in the `code` folder, as we will be referring to them. You can write CUDA code, and I can help you run it using `leetgpu`, analyze the output, and debug any issues. As a part of the workflow, before pushing any new `.cu` file, I will ensure that the filename is the same as the one mentioned in the first line of the file. If they differ, I will automatically rename the file to match the name in the file content.

## How I Can Help

I can assist you in the following ways:

*   **Code Explanations:** When asked to explain code, always provide an ELI5 (Explain Like I'm 5) explanation.
*   **Code Writing and Explanation:** I can write new CUDA kernels, explain existing code, and demonstrate various CUDA concepts like memory management, streams, and parallel programming patterns.
*   **Running and Analyzing Code:** You can ask me to run any of your `.cu` files. I will use the `leetgpu` command, show you the output, and help you interpret the results, whether it's a successful run or an error.
*   **Debugging and Optimization:** If your code has bugs or is not performing as expected, I can help you debug it. I can also suggest optimizations to improve performance, such as using shared memory or improving memory access patterns.
*   **Answering Questions:** Feel free to ask me any questions about CUDA, `leetgpu`, or GPU programming in general.

## Example Interactions

Here are a few examples of how you can interact with me:

*   **To run a file:** "run `code/4.cu`" (which will be executed as `leetgpu run code/4.cu`)
*   **To get an explanation of a concept:** "Explain shared memory in CUDA."
*   **To write a new kernel:** "Write a CUDA kernel to perform matrix multiplication."
*   **To debug an error:** "I'm getting an error in `code/5.cu`. Can you help me fix it?"

I look forward to working with you on this project! Let's get started.
