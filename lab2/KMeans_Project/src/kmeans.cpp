#include "kmeans.h"
#include "distance.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <unordered_set>

static unsigned long int next = 1;
static const unsigned long kmeans_rmax = 32767;

void kmeans_srand(unsigned int seed) {
    next = seed;
}

int kmeans_rand() {
    next = next * 1103515245 + 12345;
    return (unsigned int)(next / 65536) % (kmeans_rmax + 1);
}

std::vector<std::vector<double>> read_input_file(const std::string &filename, int dims) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<std::vector<double>> data;
    std::string line;

    // Skip the first line
    std::getline(infile, line);
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::vector<double> point(dims);

        double id;
        iss >> id;

        for (int d = 0; d < dims; ++d) {
            iss >> point[d];
        }
        data.push_back(point);
    }
    infile.close();
    return data;
}

void initialize_centroids(int k, const std::vector<std::vector<double>>& data,
                          std::vector<std::vector<double>>& centroids, int seed) {
    int n_points = data.size();
    kmeans_srand(seed);

    if (k > n_points) {
        std::cerr << "Error: The number of centroids (k) cannot be greater than the number of data points." << std::endl;
        exit(EXIT_FAILURE);
    }

    centroids.reserve(k);
    std::unordered_set<int> chosen_indices;

    while (centroids.size() < k) {
        int index = kmeans_rand() % n_points;
        if (chosen_indices.find(index) == chosen_indices.end()) {
            centroids.push_back(data[index]);
            chosen_indices.insert(index);
        }
    }
}

void run_kmeans(int k, int dims, int max_iters, double threshold, bool output_centroids, int seed,
                bool use_cuda_gmem, bool use_cuda_shmem, bool use_cuda_thrust, bool use_cpu, const std::string &input_file) {

    std::vector<std::vector<double>> data = read_input_file(input_file, dims);
    int n_points = data.size();

    std::vector<int> labels(n_points, -1);
    std::vector<std::vector<double>> centroids;
    initialize_centroids(k, data, centroids, seed);

    auto start = std::chrono::high_resolution_clock::now();
    int actual_iters = 0; 
    if (use_cuda_gmem) {
        kmeans_cuda_gmem(k, dims, max_iters, threshold, data, labels, centroids, actual_iters);
    } else if (use_cuda_shmem) {
        kmeans_cuda_shmem(k, dims, max_iters, threshold, data, labels, centroids, actual_iters);
    } else if (use_cuda_thrust) {
        kmeans_cuda_thrust(k, dims, max_iters, threshold, data, labels, centroids, actual_iters);
    } else if (use_cpu) {
        kmeans_cpu(k, dims, max_iters, threshold, data, labels, centroids, actual_iters);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time = end - start;
    double total_time = elapsed_time.count();
    double time_per_iter = total_time / actual_iters;

    // Output in the required format
    printf("%d,%.6lf\n", actual_iters, time_per_iter);

    if (output_centroids) {
        for (int cluster_id = 0; cluster_id < k; ++cluster_id) {
            printf("%d ", cluster_id);
            for (int d = 0; d < dims; ++d) {
                printf("%.8lf ", centroids[cluster_id][d]);
            }
            printf("\n");
        }
    } else {
        printf("clusters:");
        for (int i = 0; i < n_points; ++i) {
            printf(" %d", labels[i]);
        }
        printf("\n");
    }
}

void run_kmeans_from_cli(int argc, char *argv[]) {
    int k = -1, dims = -1, max_iters = 150;
    double threshold = 1e-5;
    std::string input_file;
    bool output_centroids = false;
    int seed = 8675309;
    bool use_cuda_gmem = false;
    bool use_cuda_shmem = false;
    bool use_cuda_thrust = false;
    bool use_cpu = false;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-k") {
            k = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "-d") {
            dims = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "-i") {
            input_file = argv[++i];
        } else if (std::string(argv[i]) == "-m") {
            max_iters = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "-t") {
            threshold = std::atof(argv[++i]);
        } else if (std::string(argv[i]) == "-c") {
            output_centroids = true;
        } else if (std::string(argv[i]) == "-s") {
            seed = std::atoi(argv[++i]);
        } else if (std::string(argv[i]) == "--use_cuda_gmem") {
            use_cuda_gmem = true;
        } else if (std::string(argv[i]) == "--use_cuda_shmem") {
            use_cuda_shmem = true;
        } else if (std::string(argv[i]) == "--use_cuda_thrust") {
            use_cuda_thrust = true;
        } else if (std::string(argv[i]) == "--use_cpu") {
            use_cpu = true;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    run_kmeans(k, dims, max_iters, threshold, output_centroids, seed, use_cuda_gmem, use_cuda_shmem, use_cuda_thrust, use_cpu, input_file);
}

int main(int argc, char *argv[]) {
    run_kmeans_from_cli(argc, argv);
    return 0;
}
