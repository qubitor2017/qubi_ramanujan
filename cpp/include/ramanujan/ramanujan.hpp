#pragma once
#include <string>
#include <vector>

namespace ramanujan
{
    struct LinearResult
    {
        std::vector<double> x;
        double residual{};
        double relative_residual{};
        std::string status{"solved"};
    };

    // Dense square solve using Gaussian elimination with partial pivoting.
    // For large/sparse production systems, use the Scientific module's sparse backend.
    LinearResult solve_linear(std::vector<std::vector<double>> A,
                              std::vector<double> b,
                              double relative_pivot_tolerance = 1e-12);

    double dot(const std::vector<double> &a, const std::vector<double> &b);
}
