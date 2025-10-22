// Mental Model 4.0

// ===================================
// Host (CPU) Code
// ===================================
main() {
  // Define a grid of work
  gridDim = ...;
  blockDim = ...;

  // LAUNCH: The CUDA runtime maps Blocks from the Grid onto physical SMs.
  my_kernel<<<gridDim, blockDim>>>(...);
}

// ===================================
// Device (GPU) Code
// ===================================
__global__ void my_kernel(...) {
  // EXECUTION: On an SM, threads are grouped into Warps of 32.
  // A Warp Scheduler picks a ready warp to execute.

  // INSTRUCTION: All 32 threads in the chosen warp execute this instruction.
  int idx = ...; // (e.g., ADD, MUL, etc.)

  // DIVERGENCE: If threads in a warp disagree on this 'if',
  // the paths are serialized, wasting execution slots.
  if (condition) { ... } else { ... }

  // STALL: If this instruction needs data from VRAM, the warp is paused.
  // The SM scheduler instantly picks another ready warp to execute, hiding latency.
  value = global_memory[idx];
}