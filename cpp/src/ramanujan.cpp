#include "ramanujan/ramanujan.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ramanujan
{

    double dot(const std::vector<double> &a, const std::vector<double> &b)
    {
        if (a.size() != b.size())
            throw std::invalid_argument("dot: dimension mismatch");
        double sum = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i)
            sum += a[i] * b[i];
        return sum;
    }

    LinearResult solve_linear(std::vector<std::vector<double>> A,
                              std::vector<double> b,
                              double relative_pivot_tolerance)
    {
        const std::size_t n = A.size();
        if (n == 0 || b.size() != n)
            throw std::invalid_argument("solve_linear: dimension mismatch");
        if (!(relative_pivot_tolerance > 0.0) || !std::isfinite(relative_pivot_tolerance))
            throw std::invalid_argument("solve_linear: pivot tolerance must be finite and positive");

        double matrix_scale = 0.0;
        for (const auto &row : A)
        {
            if (row.size() != n)
                throw std::invalid_argument("solve_linear: square matrix required");
            for (double value : row)
            {
                if (!std::isfinite(value))
                    throw std::invalid_argument("solve_linear: matrix must be finite");
                matrix_scale = std::max(matrix_scale, std::abs(value));
            }
        }
        for (double value : b)
            if (!std::isfinite(value))
                throw std::invalid_argument("solve_linear: right-hand side must be finite");
        if (matrix_scale == 0.0)
            throw std::runtime_error("solve_linear: singular zero matrix");

        const double pivot_threshold = relative_pivot_tolerance * matrix_scale;
        const auto A_original = A;
        const auto b_original = b;

        for (std::size_t k = 0; k < n; ++k)
        {
            std::size_t pivot = k;
            for (std::size_t i = k + 1; i < n; ++i)
                if (std::abs(A[i][k]) > std::abs(A[pivot][k]))
                    pivot = i;
            if (std::abs(A[pivot][k]) <= pivot_threshold)
                throw std::runtime_error("solve_linear: singular or numerically rank-deficient matrix at pivot " + std::to_string(k));
            std::swap(A[k], A[pivot]);
            std::swap(b[k], b[pivot]);

            const double diagonal = A[k][k];
            for (std::size_t j = k; j < n; ++j)
                A[k][j] /= diagonal;
            b[k] /= diagonal;
            for (std::size_t i = 0; i < n; ++i)
            {
                if (i == k)
                    continue;
                const double factor = A[i][k];
                for (std::size_t j = k; j < n; ++j)
                    A[i][j] -= factor * A[k][j];
                b[i] -= factor * b[k];
            }
        }

        double residual_sq = 0.0, rhs_sq = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            double value = -b_original[i];
            for (std::size_t j = 0; j < n; ++j)
                value += A_original[i][j] * b[j];
            residual_sq += value * value;
            rhs_sq += b_original[i] * b_original[i];
        }
        const double residual = std::sqrt(residual_sq);
        const double relative = residual / std::max(std::sqrt(rhs_sq), 1.0);
        return {b, residual, relative, "solved"};
    }
}
