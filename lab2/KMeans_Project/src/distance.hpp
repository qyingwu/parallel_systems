#ifndef DISTANCE_HPP
#define DISTANCE_HPP

#include <cmath>
#include <vector>

// calculate euclidean distance, helper function
inline double euclidean_distance(const std::vector<double> &point1, const std::vector<double> &point2) {
    double sum = 0.0;
    for (size_t i = 0; i < point1.size(); ++i) {
        sum += (point1[i] - point2[i]) * (point1[i] - point2[i]);
    }
    return std::sqrt(sum);
}

#endif // DISTANCE_HPP