// Mental Model 1.0

// ===================================
// Host (CPU) Code
// ===================================
main() {
  // 1. Setup, data loading, control flow
  // (e.g., loading a model, tokenizing a prompt)
  host_data = prepare_data();

  // 2. Transfer necessary data to the GPU
  device_data = send_to_gpu(host_data);

  // 3. Launch GPU work
  launch_kernel(device_data);

  // 4. Retrieve results from the GPU
  host_results = get_from_gpu(device_data);

  // 5. Final processing, output
  // (e.g., de-tokenizing, returning response)
  process_results(host_results);
}


// ===================================
// Device (GPU) Code
// ===================================
kernel(device_data) {
  // Massively parallel computation happens here.
  // Thousands of threads execute this code on their
  // slice of the 'device_data'.
}