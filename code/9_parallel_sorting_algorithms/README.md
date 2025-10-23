# Parallel Merge and Radix Sort in CUDA

This directory contains implementations for advanced parallel algorithms focusing on merging sorted arrays and radix sorting. These examples demonstrate sophisticated data-dependent algorithms that require threads to inspect input data to determine their work assignments.

## Files
- `merge.cu`: Simplified single-block parallel merge using co-rank algorithm
- `radixSort.cu`: Single-pass, single-block Radix Sort based on the least significant bit (LSB), using parallel scan
- `mm12.md`: Mental model for parallel merge operations
- `mm13.md`: Mental model for radix sort algorithm
- `concepts12.md`: Key concepts for parallel merge
- `concepts13.md`: Key concepts for radix sort

## Compilation and Execution

To compile the merge CUDA code:
```bash
nvcc -o merge merge.cu
```

To run the merge implementation:
```bash
./merge
```

To compile the radix sort CUDA code:
```bash
nvcc -o radixSort radixSort.cu
```

To run the radix sort implementation:
```bash
./radixSort
```

## Key Concepts

### Parallel Merge Algorithm
- **Co-Rank Strategy**: Each thread determines the rank of its element in the other array to find its final position
- **Binary Search**: Using binary search to efficiently determine ranks of elements in the other array
- **Data-Dependent Work Assignment**: Threads inspect input data to determine their specific work ranges
- **Sequential Sub-problems**: After determining work ranges, threads can solve sub-problems sequentially
- **Synchronization Requirements**: Using `__syncthreads()` to coordinate work between different phases

### Radix Sort Algorithm
- **Least Significant Bit (LSB) Sorting**: Sorting based on the least significant bit first
- **Bucket Partitioning**: Separating elements into buckets based on bit values (0s and 1s)
- **Parallel Scan Integration**: Using scan operations to determine final positions in sorted output
- **Exclusive vs. Inclusive Scan**: Converting between scan types to determine bucket positions
- **In-Block Sorting**: Performing local operations within a single thread block

### Data-Dependent Algorithms
- **Work Assignment**: How threads determine their work based on input data characteristics
- **Search Operations**: Using binary search and other techniques to locate relevant data ranges
- **Dynamic Load Balancing**: How workload varies based on input data distribution
- **Algorithm Complexity**: The impact of data-dependent behavior on performance

### Memory and Synchronization Patterns
- **Shared Memory Usage**: Leveraging fast block-level memory for intermediate computations
- **Synchronization Strategies**: Using `__syncthreads()` to coordinate parallel operations
- **Race Condition Prevention**: Ensuring proper ordering of reads and writes in shared memory
- **Memory Access Optimization**: Managing access patterns during multi-phase algorithms

## Design Considerations and Trade-offs

### Merge Algorithm Design
- **Search vs. Processing Trade-off**: Balancing time spent searching for ranks vs. actual merging
- **Block Size Constraints**: Limited to single-block implementation for simplicity
- **Memory Requirements**: Shared memory usage for temporary storage
- **Scalability Limitations**: Current implementation only handles arrays up to block size

### Radix Sort Design
- **Bit-Precision Limitations**: Currently only handles LSB sorting for simplicity
- **In-Block vs. Multi-Block**: Trade-offs between single-block implementation and multi-block complexity
- **Memory Pattern Optimization**: Ensuring optimal memory access during bucketing and scan operations
- **Numerical Constraints**: Limited to integer sorting with known bit width

### Implementation Robustness
- **Boundary Condition Handling**: Proper management of elements at array boundaries
- **Synchronization Requirements**: Ensuring correctness with proper barrier synchronization
- **Verification Strategies**: Comparing results with sequential CPU implementations
- **Error Handling**: Proper CUDA API error checking for all operations

## Performance Considerations
- **Memory Bandwidth**: Both operations can be memory-bound, requiring optimization of access patterns
- **Arithmetic Intensity**: The ratio of computation to memory access varies with implementation
- **Cache Effectiveness**: How efficiently the algorithm uses the GPU's memory hierarchy
- **Load Balancing**: Managing variable computation requirements across threads

## Modern AI/ML Context

### PyTorch, JAX, & MLX Connection
- PyTorch's sorting operations like `torch.sort()` use optimized parallel algorithms
- JAX's `lax.sort` provides functional sorting operations that map to GPU implementations
- Database and data processing libraries use parallel sorting and merging for large-scale operations
- Libraries like cuDF implement these concepts for GPU-accelerated data frame operations

### LLM Training and Inference Relevance
- **Attention Mechanisms**: Sorting operations can be used in sparse attention mechanisms
- **Data Processing**: Large-scale dataset preprocessing requires efficient sorting algorithms
- **Indexing Operations**: Sorting used in various indexing and retrieval operations
- **Dynamic Batching**: Sorting by sequence length for efficient dynamic batching

### Bottleneck Analysis
Understanding parallel merge and radix sort is critical for addressing bottlenecks in AI/ML:
- **Memory Bottleneck**: Efficient memory access patterns are crucial for performance in both operations
- **Compute Bottleneck**: The arithmetic intensity affects how efficiently compute resources are used
- **Synchronization Bottleneck**: The multiple `__syncthreads()` calls in these algorithms can impact performance
- **Scalability Bottleneck**: Ensuring algorithms scale with increasing data size and hardware resources

## Implementation Details

### Parallel Merge Implementation
The parallel merge implementation demonstrates:
1. Use of binary search to find element ranks in the other array
2. Co-rank strategy where each thread determines its element's final position
3. Two-phase approach: first for elements from array A, then for elements from array B
4. Synchronization to coordinate between phases
5. CPU verification using `std::merge` to ensure correctness

### Radix Sort Implementation
The radix sort implementation demonstrates:
1. Bit extraction for determining element buckets (0s and 1s)
2. Parallel scan for calculating bucket positions in the output array
3. Conversion between inclusive and exclusive scan for proper positioning
4. In-block sorting using shared memory for temporary storage
5. CPU verification using manual sorting to ensure correctness

Both implementations showcase important concepts in data-dependent parallel algorithms, where threads must inspect input data to determine their specific computational tasks.