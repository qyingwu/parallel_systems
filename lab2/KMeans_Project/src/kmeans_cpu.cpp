#include "kmeans.h"
#include "distance.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <ctime>

// sequential CPU-based KMeans function
void kmeans_cpu(int k, int dims, int max_iters, double threshold, const std::vector<std::vector<double>> &data, 
                std::vector<int> &labels, std::vector<std::vector<double>> &centroids, int &iterations_run) {

    int n_points = data.size();
    centroids.clear();

    // Initialize centroids randomly using initialize_centroids
    initialize_centroids(k, data, centroids, time(0));

    std::vector<std::vector<double>> new_centroids(k, std::vector<double>(dims, 0.0));
    std::vector<int> cluster_sizes(k, 0);

    double total_shift = 0.0;

    iterations_run = 0; 

    for (int iter = 0; iter < max_iters; ++iter) {
        total_shift = 0.0;
        iterations_run++;

        // Step 1: Assign each point to the nearest centroid
        for (int i = 0; i < n_points; ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int closest_centroid = -1;

            for (int c = 0; c < k; ++c) {
                double dist = euclidean_distance(data[i], centroids[c]);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_centroid = c;
                }
            }
            labels[i] = closest_centroid;
        }

        // Step 2: Recalculate the centroids based on the assignments
        new_centroids.assign(k, std::vector<double>(dims, 0.0));
        cluster_sizes.assign(k, 0);

        for (int i = 0; i < n_points; ++i) {
            int cluster_id = labels[i];
            for (int d = 0; d < dims; ++d) {
                new_centroids[cluster_id][d] += data[i][d];
            }
            cluster_sizes[cluster_id]++;
        }

        // Normalize the new centroids by dividing by the number of points assigned to each cluster
        for (int c = 0; c < k; ++c) {
            if (cluster_sizes[c] > 0) {
                for (int d = 0; d < dims; ++d) {
                    new_centroids[c][d] /= cluster_sizes[c];
                }
            } else {
                // Reinitialize centroid using the initialize_centroids function if no points were assigned
                std::cerr << "Cluster " << c << " is empty. Reinitializing centroid..." << std::endl;
                std::vector<std::vector<double>> temp_centroids;
                initialize_centroids(1, data, temp_centroids, time(0));  // Reinitialize one centroid
                new_centroids[c] = temp_centroids[0];  // Assign the new randomly initialized centroid
            }
        }

        // Step 3: Check convergence by calculating the shift of centroids
        total_shift = 0.0;
        for (int c = 0; c < k; ++c) {
            total_shift += euclidean_distance(centroids[c], new_centroids[c]);
        }

        centroids = new_centroids;

        // Print shift for debugging
        std::cout << "Iteration " << iter + 1 << ": Total Shift = " << total_shift << std::endl;

        // Stop if the total shift is smaller than the threshold
        if (total_shift <= threshold) {
            std::cout << "Converged in " << iter + 1 << " iterations." << std::endl;
            break;
        }
    }

    // Output final centroids
    std::cout << "Final centroids:" << std::endl;
    for (int c = 0; c < k; ++c) {
        std::cout << "Centroid " << c << ": ";
        for (int d = 0; d < dims; ++d) {
            std::cout << centroids[c][d] << " ";
        }
        std::cout << std::endl;
    }
}
