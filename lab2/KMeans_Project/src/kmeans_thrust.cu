#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>
#include "kmeans.h"
#include "atomic_utils.h"


// Functor to calculate squared Euclidean distance
struct SquaredDistance {
    const double* points;
    const double* centroids;
    int dims;

    SquaredDistance(const double* p, const double* c, int d) : points(p), centroids(c), dims(d) {}

    __device__
    double operator()(int point_idx, int centroid_idx) const {
        double dist = 0.0;
        for (int d = 0; d < dims; ++d) {
            double diff = points[point_idx * dims + d] - centroids[centroid_idx * dims + d];
            dist += diff * diff;
        }
        return dist;
    }
};



// CUDA kernel to assign points to the nearest centroids
__global__ void assign_points_to_centroids(const double* points, const double* centroids, int* labels, int n_points, int n_centroids, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points) {
        double min_dist = INFINITY;
        int best_centroid = 0;

        for (int c = 0; c < n_centroids; ++c) {
            double dist = 0.0;
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
__global__ void compute_new_centroids(const double* points, const int* labels, double* centroids, int* counts, int n_points, int n_centroids, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_points) {
        int centroid_idx = labels[idx];
        for (int d = 0; d < dims; ++d) {
            atomicAdd(&centroids[centroid_idx * dims + d], points[idx * dims + d]);
        }
        atomicAdd(&counts[centroid_idx], 1);
    }
}

// CUDA kernel to normalize centroids
__global__ void normalize_centroids(double* centroids, const int* counts, int n_centroids, int dims) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_centroids) {
        for (int d = 0; d < dims; ++d) {
            centroids[idx * dims + d] /= counts[idx];
        }
    }
}




// Function to run KMeans with CUDA kernels
void kmeans_thrust(int k, int dims, int max_iters, double threshold,
                   const std::vector<std::vector<double>>& data, 
                   std::vector<int>& labels, std::vector<std::vector<double>>& centroids) {

    int n_points = data.size();

    // Copy data to device
    thrust::host_vector<double> h_points(n_points * dims);
    for (int i = 0; i < n_points; ++i) {
        for (int d = 0; d < dims; ++d) {
            h_points[i * dims + d] = data[i][d];
        }
    }
    thrust::device_vector<double> d_points = h_points;

    // Initialize centroids
    initialize_centroids(k, data, centroids, time(0));

    thrust::host_vector<double> h_centroids(k * dims);
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < dims; ++d) {
            h_centroids[i * dims + d] = centroids[i][d];
        }
    }
    thrust::device_vector<double> d_centroids = h_centroids;

    thrust::device_vector<int> d_labels(n_points);
    thrust::device_vector<int> d_counts(k);

    for (int iter = 0; iter < max_iters; ++iter) {
        int blocks = (n_points + 255) / 256;

        // Assign points to centroids
        assign_points_to_centroids<<<blocks, 256>>>(thrust::raw_pointer_cast(d_points.data()), thrust::raw_pointer_cast(d_centroids.data()), thrust::raw_pointer_cast(d_labels.data()), n_points, k, dims);
        cudaDeviceSynchronize();

        // Reset centroids and counts
        thrust::fill(d_centroids.begin(), d_centroids.end(), 0.0);
        thrust::fill(d_counts.begin(), d_counts.end(), 0);

        // Compute new centroids
        compute_new_centroids<<<blocks, 256>>>(thrust::raw_pointer_cast(d_points.data()), thrust::raw_pointer_cast(d_labels.data()), thrust::raw_pointer_cast(d_centroids.data()), thrust::raw_pointer_cast(d_counts.data()), n_points, k, dims);
        cudaDeviceSynchronize();

        // Normalize centroids
        int norm_blocks = (k + 255) / 256;
        normalize_centroids<<<norm_blocks, 256>>>(thrust::raw_pointer_cast(d_centroids.data()), thrust::raw_pointer_cast(d_counts.data()), k, dims);
        cudaDeviceSynchronize();
    }

    // Copy the final centroids and labels back to the host
    thrust::copy(d_centroids.begin(), d_centroids.end(), h_centroids.begin());
    for (int i = 0; i < k; ++i) {
        for (int d = 0; d < dims; ++d) {
            centroids[i][d] = h_centroids[i * dims + d];
        }
    }
    thrust::copy(d_labels.begin(), d_labels.end(), labels.begin());
}