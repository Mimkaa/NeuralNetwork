#pragma once
#include <vector>
#include <Eigen/Dense>

struct WeightsBiasesReturnStructure {
    std::vector<Eigen::MatrixXd> gradientsWRTWeights;
    Eigen::VectorXd gradientsWRTBiases;

    // Constructor to initialize the gradients with the given size
    WeightsBiasesReturnStructure(size_t size)
        : gradientsWRTWeights(size),
        gradientsWRTBiases(Eigen::VectorXd::Zero(size)) {}
};
