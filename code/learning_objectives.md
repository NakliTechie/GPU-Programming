### Part I: Fundamental Concepts

**Chapter 1: Introduction**
*   **Learning Objectives:**
    *   Explain the fundamental motivation for parallel computing and the limitations of sequential processing.
    *   Differentiate between the design philosophies of CPUs (latency-oriented) and GPUs (throughput-oriented).
    *   Define heterogeneous computing.
    *   Identify the primary challenges in parallel programming, such as managing complexity and achieving high performance.

**Chapter 2: Heterogeneous Data Parallel Computing**
*   **Learning Objectives:**
    *   Define data parallelism and identify it in simple problems.
    *   Describe the basic structure of a CUDA C program, distinguishing between host (CPU) and device (GPU) code.
    *   Write a simple `__global__` kernel function.
    *   Manage device memory: allocate (`cudaMalloc`), deallocate (`cudaFree`), and transfer data between host and device (`cudaMemcpy`).
    *   Explain the basic CUDA threading model: threads, blocks, and grids.
    *   Launch a kernel using the `<<<...>>>` syntax and calculate thread indices.

**Chapter 3: Multidimensional Grids and Data**
*   **Learning Objectives:**
    *   Organize CUDA threads and blocks into 2D or 3D grids to naturally map to multidimensional problems.
    *   Write kernels that map multidimensional thread and block indices to data elements in arrays.
    *   Explain and implement linearization for accessing multidimensional data stored in row-major layout.
    *   Implement a naive parallel matrix multiplication kernel.

**Chapter 4: Compute Architecture and Scheduling**
*   **Learning Objectives:**
    *   Describe the high-level hardware architecture of a modern GPU, focusing on the role of Streaming Multiprocessors (SMs).
    *   Explain how thread blocks are scheduled onto SMs and the concept of "transparent scalability."
    *   Define a **warp** as the fundamental unit of thread scheduling and its relation to SIMD (Single Instruction, Multiple Data) execution.
    *   Analyze the performance impact of **control divergence** within a warp.
    *   Explain how massive multithreading is used by the GPU to hide memory latency.

**Chapter 5: Memory Architecture and Data Locality**
*   **Learning Objectives:**
    *   Differentiate between the various levels of the CUDA memory hierarchy: registers, shared memory, and global memory.
    *   Analyze the importance of memory access efficiency and the concept of being "memory-bound."
    *   Implement the **tiling** technique using on-chip shared memory to reduce global memory traffic.
    *   Write a high-performance tiled matrix multiplication kernel that leverages shared memory for data locality.

**Chapter 6: Performance Considerations**
*   **Learning Objectives:**
    *   Define and implement **memory coalescing** to maximize the utilization of global memory bandwidth.
    *   Explain how having a high number of active warps (high occupancy) helps hide memory latency.
    *   Implement **thread coarsening**, where a single thread performs more work to reduce redundant operations or overhead.
    *   Synthesize a checklist of key optimization strategies for CUDA programs.

### Part II: Parallel Patterns

**Chapter 7: Convolution**
*   **Learning Objectives:**
    *   Implement a parallel 2D convolution algorithm, a common pattern in image processing and deep learning.
    *   Utilize **constant memory** and its associated cache to optimize broadcast, read-only data access (e.g., for filter kernels).

**Chapter 8: Stencil**
*   **Learning Objectives:**
    *   Implement parallel stencil computations, a core pattern in scientific simulations on grids.
    *   Apply shared memory tiling to 3D stencil problems.
    *   Introduce **register tiling** as a further optimization to reduce shared memory pressure.

**Chapter 9: Parallel Histogram**
*   **Learning Objectives:**
    *   Use **atomic operations** to safely manage race conditions where multiple threads update the same memory location.
    *   Implement a basic parallel histogram using atomics.
    *   Apply **privatization** (using local, private histograms) to reduce contention and improve the performance of atomic operations.

**Chapter 10: Reduction**
*   **Learning Objectives:**
    *   Implement a parallel reduction (e.g., sum, max) using a reduction tree pattern.
    *   Analyze and apply strategies to minimize control divergence and memory divergence within a reduction kernel.

**Chapter 11: Prefix Sum (Scan)**
*   **Learning Objectives:**
    *   Explain the importance of the prefix sum (scan) operation in parallelizing sequential algorithms.
    *   Implement a work-inefficient but highly parallel scan algorithm (Kogge-Stone).
    *   Implement a more work-efficient but less parallel scan algorithm (Brent-Kung).
    *   Analyze the critical trade-off between speed (parallelism) and work efficiency.

**Chapter 12: Merge**
*   **Learning Objectives:**
    *   Implement a parallel merge algorithm for combining two sorted lists.
    *   Understand and implement the **co-rank function**, which allows threads to dynamically identify their input data range in data-dependent algorithms.

### Part III: Advanced Patterns and Applications

**Chapter 13: Sorting**
*   **Learning Objectives:**
    *   Implement a parallel non-comparison-based sort: **radix sort**.
    *   Implement a parallel comparison-based sort: **merge sort**, building upon the merge pattern.

**Chapter 14: Sparse Matrix Computation**
*   **Learning Objectives:**
    *   Implement Sparse Matrix-Vector multiplication (SpMV), a key kernel in scientific and graph analytics.
    *   Compare and contrast the performance implications of different sparse matrix storage formats (COO, CSR, ELL).

**Chapter 15: Graph Traversal**
*   **Learning Objectives:**
    *   Implement a parallel Breadth-First Search (BFS) algorithm.
    *   Compare vertex-centric and edge-centric parallelization strategies and their respective trade-offs in terms of load balancing and parallelism.

**Chapter 16: Deep Learning**
*   **Learning Objectives:**
    *   Understand the fundamental computational kernels of Convolutional Neural Networks (CNNs).
    *   Explain how to formulate a convolutional layer as a General Matrix-Matrix Multiplication (GEMM) operation to leverage highly optimized libraries.

**Chapter 17-19 (Case Studies & Synthesis)**
*   **Learning Objectives:**
    *   Apply the fundamental patterns and optimization techniques to large, real-world scientific applications (MRI, Electrostatic Potential Maps).
    *   Synthesize the concepts of algorithm selection and problem decomposition into a general "computational thinking" framework for tackling new parallel problems.

### Part IV: Advanced Practices

**Chapter 20: Programming a Heterogeneous Computing Cluster**
*   **Learning Objectives:**
    *   Use **CUDA Streams** to overlap data transfers with kernel execution, hiding communication latency.
    *   Understand the basic model for programming a multi-node cluster using the Message Passing Interface (MPI) in conjunction with CUDA.

**Chapter 21: CUDA Dynamic Parallelism**
*   **Learning Objectives:**
    *   Use dynamic parallelism to enable a CUDA kernel to launch other kernels on the GPU.
    *   Implement recursive algorithms, such as quadtree construction, on the GPU without returning control to the host.

**Chapter 22: Advanced Practices and Future Evolution**
*   **Learning Objectives:**
    *   Describe advanced host/device interaction models, including **Unified Memory** and its benefits for programmer productivity.
    *   Understand the evolution of kernel execution control, including function calls within kernels and dynamic parallelism.

**Chapter 23: Conclusion and Outlook**
*   **Learning Objectives:**
    *   Recap the key goals and techniques of massively parallel programming.
    *   Reflect on the future trends and evolution of the parallel computing landscape.