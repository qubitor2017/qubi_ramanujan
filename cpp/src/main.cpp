#include "ramanujan/ramanujan.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

int main()
{
    auto r = ramanujan::solve_linear({{2, 1}, {1, 3}}, {5, 6});
    if (std::abs(r.x[0] - 1.8) > 1e-9 || std::abs(r.x[1] - 1.4) > 1e-9)
        return 1;
    if (r.residual > 1e-10 || r.relative_residual > 1e-10 || r.status != "solved")
        return 2;
    bool rejected = false;
    try
    {
        (void)ramanujan::solve_linear({{1, 2}, {2, 4}}, {1, 2});
    }
    catch (const std::runtime_error &)
    {
        rejected = true;
    }
    if (!rejected)
        return 3;
    std::cout << "AI_Ramanujan C++ smoke and singularity tests PASS\n";
    return 0;
}
