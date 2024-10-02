#include "kmeans.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <chrono>

// Main function
int main(int argc, char *argv[]) {
    // Command-line argument parsing
    int k = -1, dims = -1, max_iters = 150;
    double threshold = 1e-4;
    std::string input_file;
    bool output_centroids = false;
    int seed = 8675309;  // Default seed for random initialization
    bool use_cuda = false;
    bool use_thrust = false;

    // Parsing command-line arguments
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
        } else if (std::string(argv[i]) == "--use_cuda") {
            use_cuda = true;
        } else if (std::string(argv[i]) == "--use_thrust") {
            use_thrust = true;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            return EXIT_FAILURE;
        }
    }

    // Validate mandatory arguments
    if (k == -1 || dims == -1 || input_file.empty()) {
        std::cerr << "Usage: " << argv[0] << " -k num_clusters -d dims -i input_file [-m max_iters] [-t threshold] [-c] [-s seed] [--use_cuda] [--use_thrust]" << std::endl;
        return EXIT_FAILURE;
    }

    // Start measuring the time
    auto start_time = std::chrono::high_resolution_clock::now();

    // Call run_kmeans to run the chosen KMeans implementation
    run_kmeans(k, dims, max_iters, threshold, output_centroids, seed, use_cuda, use_thrust, input_file);

    // End measuring time
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;

    // Calculate time per iteration
    double time_per_iteration = elapsed_time.count() / max_iters;

    // Output time per iteration and number of iterations
    std::cout << max_iters << "," << time_per_iteration << std::endl;

    return 0;
}
