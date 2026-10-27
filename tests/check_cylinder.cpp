// check_cylinder.cpp — headless sanity check of the CPU solver
// i.e. im just looking for values, no pretty window for this one :(
//
// Build + run (from C:\Users\diego\lbm):
//   g++ -O3 -march=native -fopenmp tests/check_cylinder.cpp -o check_cylinder.exe
//   ./check_cylinder.exe
//
// Runs the same cylinder flow as the viewer for 40k steps and checks:
//   1. no blow-up (density stays near 1 -> weakly compressible, as expected) -> LBM DOESNT DO COMPRESSIBLE TOO MUCH
//   2. vortex shedding happens (probe behind cylinder oscillates)
//   3. Strouhal number St = f D / U is in the right range (~0.16-0.21 at Re=100)

#include "../src/lbm_core.h"
#include <algorithm>
#include <chrono>
#include <cstdio>

// same setup as the viewer
constexpr int NX = 400, NY = 100;
constexpr real U0 = 0.1, RE = 100;
constexpr int CYL_X = NX / 5, CYL_Y = NY / 2, CYL_R = NY / 9;

int main() {
    Lattice L(NX, NY);
    L.setViscosity(U0 * 2 * CYL_R / RE);
    setupFlowPast(L, U0, [](int x, int y) {
        int dx = x - CYL_X, dy = y - CYL_Y;
        return dx * dx + dy * dy <= CYL_R * CYL_R;
    });
    kickWake(L, CYL_X + CYL_R, CYL_Y, CYL_R, 0.1 * U0);  // start the shedding (see lbm_core.h)

    const int STEPS = 40000, WARMUP = 20000;  // first 20k steps: let shedding develop
    const int px = CYL_X + 4 * CYL_R, py = CYL_Y;  // probe 4 radii behind the cylinder
    double vmin = 1e9, vmax = -1e9, rmin = 1e9, rmax = -1e9, prev = 0;
    int periods = 0;

    auto t0 = std::chrono::steady_clock::now();
    for (int t = 1; t <= STEPS; t++) {
        L.step();
        double r = L.rho[L.idx(NX / 2, NY / 2)];
        if (std::isnan(r)) { std::printf("FAIL: diverged (NaN) at step %d\n", t); return 1; }
        if (t > WARMUP) {
            double v = L.uy[L.idx(px, py)];
            vmin = std::min(vmin, v); vmax = std::max(vmax, v);
            rmin = std::min(rmin, r); rmax = std::max(rmax, r);
            if (prev < 0 && v >= 0) periods++;  // count upward zero crossings
            prev = v;
        }
        if (t % 10000 == 0) std::printf("  step %5d  rho=%.4f  probe uy=%+.4f\n", t, r, L.uy[L.idx(px, py)]);
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    double St = periods / double(STEPS - WARMUP) * 2 * CYL_R / U0;
    bool okRho = rmin > 0.95 && rmax < 1.05;
    bool okShed = vmax - vmin > 0.02;
    bool okSt = St > 0.15 && St < 0.22;  // a bit above 0.165 because of 22% blockage

    std::printf("\nRe=%.0f tau=%.4f  |  %.1f MLUPS\n", RE, L.tau, double(STEPS) * NX * NY / secs / 1e6);
    std::printf("[%s] density in [%.4f, %.4f]\n", okRho ? "PASS" : "FAIL", rmin, rmax);
    std::printf("[%s] shedding: probe uy in [%+.4f, %+.4f]\n", okShed ? "PASS" : "FAIL", vmin, vmax);
    std::printf("[%s] Strouhal = %.3f (expect ~0.16-0.21)\n", okSt ? "PASS" : "FAIL", St);
    return (okRho && okShed && okSt) ? 0 : 1;
}
