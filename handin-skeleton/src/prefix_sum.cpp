#include "prefix_sum.h"
#include "helpers.h"
#include <cmath>

void* compute_prefix_sum(void *a)
{
    //prefix_sum_args_t *args = (prefix_sum_args_t *)a;

    /************************
     * Your code here...    *
     * or wherever you like *
     ************************/
    
    prefix_sum_args_t *args = static_cast<prefix_sum_args_t*>(a);
    int n = args->n_vals;
    int *input = args->input_vals;
    int *output = args->output_vals;
    int thread_id = args->t_id;
    int num_threads = args->n_threads;
    int (*op)(int, int, int) = args->op;
    int n_loops = args->n_loops;

    // Hybrid approach for small inputs
    if (n <= 1024 && n_loops <= 100) {
        output[0] = input[0];
        for (int i = 1; i < n; ++i) {
            output[i] = op(output[i - 1], input[i], n_loops);
        }
        return nullptr;
    }

    // Calculate the range each thread is responsible for (static work distribution)
    int chunk_size = (n + num_threads - 1) / num_threads;
    int start = thread_id * chunk_size;
    int end = std::min(start + chunk_size, n);

    for (int i = start; i < end; ++i) {
        output[i] = input[i];
    }

    //Ensure all threads are ready before starting
    barrier_wait();  

    // Up-sweep phase with reduced barriers
    //levels of the tree in the up-sweep phase, starting from the leaves and moving up to the root
    // Up-sweep phase with reduced barriers
    int sync_interval = 2;  // Tunable synchronization interval
    for (int d = 0; d < log2(n); ++d) {
        // stride = 2^(d+1)
        int stride = 1 << (d + 1);  
        
        // Parallel loop
        for (int k = start; k < end; k += stride) {
            int left_index = k + (1 << d) - 1;        // Corresponds to x[k + 2^d - 1]
            int right_index = k + (1 << (d + 1)) - 1; // Corresponds to x[k + 2^(d+1) - 1]
            
            if (right_index < n) { // Ensure valid index
                output[right_index] = op(output[left_index], output[right_index], n_loops);
            }
        }
        
        // Reduce synchronization frequency (every 2nd iteration)
        if (d % sync_interval == 0) {
            barrier_wait();
        }
    }


    // Down-sweep phase 
    if (thread_id == 0) {
        output[n - 1] = 0;  // Set the last element to the identity element
    }
    barrier_wait();

    // level of tree up diwb-sweep phase, startging from second-to-last level and moving up tp root
    for (int d = log2(n) - 1; d >= 0; --d) {
        //distance between elements to be processed for each level
        //doubles at each level as we move up the tree.
        // 2^(d+1) for stride

        int stride = 1 << (d + 1);
        //processes elements at the current level.
        // Parallel loop to process elements
        for (int k = start; k < end; k += stride) {
            int left_child = k + (1 << d) - 1;           // Left child: x[k + 2^d - 1]
            int right_child = k + (1 << (d + 1)) - 1;    // Parent (right child): x[k + 2^(d+1) - 1]

            if (right_child < n) { 
                int temp = output[left_child];           // Step 4: Store left child's value in temp
                output[left_child] = output[right_child]; // Step 5: Replace left child with parent's value
                output[right_child] = op(temp, output[right_child], n_loops); // Step 6: Update parent (right child)
            }
        }

        // Reduce synchronization frequency
        if (d % sync_interval == 0) {
            barrier_wait();
        }
    }

    return nullptr;
}
