#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include "kmeans.h"
#include "atomic_utils.h"
#include <cmath>
#include <iostream>

// CUDA kernel to assign points to the nearest centroids
__global__ void assign_points_to_centroids_thrust(const double* points, const double* centroids, int* labels, int n_points, int n_centroids, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points) {
        double min_dist = INFINITY;
        int best_centroid = 0;

        for (int c = 0; c < n_centroids; ++c) {
            double dist = 0.0f;
            for (int d = 0; d < dims; ++d) {
                double diff = points[idx * dims + d] - centroids[c * dims + d];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_centroid = c;
            }
        }
        labels[idx] = best_centroid;
    }
}

// CUDA kernel to compute new centroids
__global__ void compute_new_centroids_thrust(const double* points, const int* labels, double* centroids, int* counts, int n_points, int n_centroids, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points) {
        int centroid_idx = labels[idx];
        atomicAdd(&counts[centroid_idx], 1);
        for (int d = 0; d < dims; ++d) {
            atomicAdd(&centroids[centroid_idx * dims + d], points[idx * dims + d]);
        }
    }
}

// Kernel to normalize the centroids by dividing by the count of assigned points
__global__ void normalize_centroids_thrust(double* centroids, const int* counts, int n_centroids, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int centroid_idx = idx / dims;
    int dim = idx % dims;
    if (centroid_idx < n_centroids && counts[centroid_idx] > 0) {
        centroids[centroid_idx * dims + dim] /= counts[centroid_idx];
    }
}

// Kernel to calculate differences between old and new centroids
__global__ void calculate_centroid_differences(const double* old_centroids, const double* new_centroids, double* differences, int k, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < k * dims) {  // Calculate for all dimensions of all centroids
        double diff = new_centroids[idx] - old_centroids[idx];
        differences[idx] = diff * diff;
    }
}

// Function to run KMeans with CUDA kernels and convergence check, with event timing
void kmeans_cuda_thrust(int k, int dims, int max_iters, double threshold,
                   const std::vector<std::vector<double>>& data, 
                   std::vector<int>& labels, std::vector<std::vector<double>>& centroids) {

    int n_points = data.size();

    // Copy data to device, points
    thrust::host_vector<double> h_points(n_points * dims);
    for (int i = 0; i < n_points; ++i) {
        for (int d = 0; d < dims; ++d) {
            h_points[i * dims + d] = data[i][d];
        }
    }
    thrust::device_vector<double> d_points = h_points;

    thrust::host_vector<double> h_centroids(k * dims);
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < dims; ++d) {
            h_centroids[i * dims + d] = centroids[i][d];
        }
    }
    thrust::device_vector<double> d_centroids = h_centroids;
    thrust::device_vector<double> d_old_centroids = d_centroids;

    thrust::device_vector<int> d_labels(n_points);
    thrust::device_vector<int> d_counts(k);

    thrust::device_vector<double> d_differences(k * dims);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);  // Start recording time

    for (int iter = 0; iter < max_iters; ++iter) {
        
        int blocks = (n_points + 255) / 256;

        // Assign points to centroids
        assign_points_to_centroids_thrust<<<blocks, 256>>>(thrust::raw_pointer_cast(d_points.data()), thrust::raw_pointer_cast(d_centroids.data()), thrust::raw_pointer_cast(d_labels.data()), n_points, k, dims);
        cudaDeviceSynchronize();

        // Swap the old and new centroids
        thrust::copy(d_centroids.begin(), d_centroids.end(), d_old_centroids.begin());

        // Reset centroids and counts
        thrust::fill(d_centroids.begin(), d_centroids.end(), 0.0f);
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        // Compute new centroids
        compute_new_centroids_thrust<<<blocks, 256>>>(thrust::raw_pointer_cast(d_points.data()), thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_centroids.data()), thrust::raw_pointer_cast(d_counts.data()), n_points, k, dims);
        cudaDeviceSynchronize();

        // Normalize centroids based on counts
        normalize_centroids_thrust<<<(k * dims + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_centroids.data()), thrust::raw_pointer_cast(d_counts.data()), k, dims);
        cudaDeviceSynchronize();

        // Calculate differences between old and new centroids
        calculate_centroid_differences<<<(k * dims + 255) / 256, 256>>>(thrust::raw_pointer_cast(d_old_centroids.data()), thrust::raw_pointer_cast(d_centroids.data()), thrust::raw_pointer_cast(d_differences.data()), k, dims);
        cudaDeviceSynchronize();

        // Per-dimension convergence check
        bool converged = true;
        std::vector<double> h_differences(k * dims);
        thrust::copy(d_differences.begin(), d_differences.end(), h_differences.begin());

        for (int i = 0; i < k; ++i) {
            for (int d = 0; d < dims; ++d) {
                double dimension_shift = sqrt(h_differences[i * dims + d]);

                if (dimension_shift > threshold) {
                    converged = false;
                    break;  // Exit early if any dimension exceeds the threshold
                }
            }

            if (!converged) break;  // Stop checking once a non-converged centroid is found
        }

        if (converged) {
            std::cout << "Convergence reached after iteration " << iter + 1 << std::endl;
            break;
        }
    }

    cudaEventRecord(stop);  // Stop recording time

    // Wait for the stop event to complete
    cudaEventSynchronize(stop);

    // Calculate elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Total kernel execution time: " << milliseconds << " ms" << std::endl;

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy the final centroids and labels back to the host
    thrust::copy(d_centroids.begin(), d_centroids.end(), h_centroids.begin());
    
    // Reshape h_centroids (1D vector) into centroids (2D vector)
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < dims; ++d) {
            centroids[i][d] = h_centroids[i * dims + d];
        }
    }

    thrust::copy(d_labels.begin(), d_labels.end(), labels.begin());

}
