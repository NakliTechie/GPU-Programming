# Code Directory 8: Parallel Reduction and Scan Operations

This directory contains implementations of fundamental parallel algorithms for data aggregation and scan operations in CUDA, demonstrating efficient reduction and prefix sum computations.

## Key Differences from Previous Examples

The reduction and scan operations are radically different from previous CUDA examples we've encountered. These operations require:

- **Active thread collaboration** rather than independent processing
- **Complex communication patterns** between threads within blocks
- **Multiple synchronization points** (`__syncthreads()`) for correctness
- **Data combination** rather than isolated element processing
- **Sophisticated parallel algorithms** that transform inherently sequential problems

While previous operations like vector addition had each thread working independently (N inputs → N independent outputs), reduction and scan require threads to actively share and combine data with each other.

## Files

### `reduction.cu` - Optimized Parallel Sum Reduction with Shared Memory
A CUDA implementation of an optimized parallel sum reduction that efficiently combines N values into a single result using shared memory for fast inter-thread communication. This implementation demonstrates:

- Efficient reduction algorithm using shared memory to minimize global memory accesses
- Power-of-two block size for optimal performance and simple indexing
- Sequential halving pattern in the reduction loop (N elements → N/2 → N/4 → ... → 1)
- Proper handling of boundary conditions when the number of elements doesn't perfectly align with block size
- Two-stage reduction process: first within blocks, then final aggregation on the CPU
- Synchronization using `__syncthreads()` to ensure all threads complete each reduction step before proceeding
- Verification by comparing GPU results with CPU computation

The reduction operation is a fundamental parallel algorithm that combines elements using an associative operation (like addition), and is different from previous examples because threads must collaborate to aggregate data rather than processing elements independently.

#### ELI5: Reduction Operation
Think of this like a big group math game where you have a million numbers (all 1.0), and you want to add them all up to find the total sum (which should be 1,048,576 since there are that many 1.0s).

Instead of one person doing all the adding, you split the work among many people (like your GPU has many tiny workers called "threads"). Here's how it works:

1. **Setup Phase**: Imagine you have 1,048,576 pieces of paper with "1.0" written on each one. You divide these papers into 4,096 groups (blocks), with 256 papers in each group.

2. **Loading Phase**: Each person in a group gets one paper with a number on it.

3. **Race Phase**: Within each group, the people start racing to add numbers together in a special pattern:
   - First, person 0 teams up with person 128 to add their numbers
   - Then person 0 teams up with person 64 to add that result
   - Then person 0 teams up with person 32 to add that result
   - And so on, halving the distance each time
   - Eventually, person 0 in each group ends up with the sum of all 256 numbers in their group

4. **Finishing Phase**: Each group now has one person (person 0) holding the sum for their group. Then someone has to add up all those group totals to get the final answer.

The program compares its answer (from the GPU) with the same math done by a regular computer processor (CPU) to make sure it got the right answer (Verification Successful!).

This is an efficient way to add millions of numbers really fast by having many tiny workers do parts of the work simultaneously, rather than just one worker doing everything in order.

### `scan.cu` - Single-Block Inclusive Scan using Kogge-Stone Algorithm
A CUDA implementation of a parallel inclusive scan (prefix sum) operation using the Kogge-Stone algorithm for a single block. This implementation demonstrates:

- Inclusive scan operation where each output element is the sum of all input elements up to and including that position
- Kogge-Stone algorithm with up-sweep phase for parallel computation
- Proper handling of power-of-two block sizes for the simple implementation
- Use of shared memory for efficient inter-thread communication
- Multiple synchronization points to ensure correct data ordering during computation
- Verification by comparing GPU results with CPU std::partial_sum implementation

The scan operation is another fundamental parallel primitive that is essential for many algorithms, differing from reduction in that it produces N outputs from N inputs rather than a single aggregate value.

#### ELI5: Scan Operation
Imagine you have a row of 16 friends, and each friend is holding a card with the number 1 written on it. A scan operation (also called a prefix sum) is like asking each friend to count how many total 1's there are from the first person up to and including themselves.

So:

- Friend 1: "I have a 1, and there are no one before me, so I say '1'"
- Friend 2: "I have a 1, and the person before me had 1, so I say '2'"
- Friend 3: "I have a 1, and the first two people had 2 total, so I say '3'"
- Friend 4: "I have a 1, and the first three people had 3 total, so I say '4'"
- And so on...
- Friend 16: "I have a 1, and the first fifteen people had 15 total, so I say '16'"

The final result is the sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16

In more technical terms, the GPU is doing an "inclusive scan" operation where:
- Output[0] = Input[0]
- Output[1] = Input[0] + Input[1]
- Output[2] = Input[0] + Input[1] + Input[2]
- Output[n] = Input[0] + Input[1] + ... + Input[n]

This is different from the reduction operation where you end up with just one number (the total sum). In a scan operation, you get back as many numbers as you put in, but each number represents a cumulative sum up to that position.

The Kogge-Stone algorithm implemented in this file does this efficiently in parallel by having each friend (thread) communicate with others in a specific pattern to calculate their cumulative sum much faster than doing it sequentially.