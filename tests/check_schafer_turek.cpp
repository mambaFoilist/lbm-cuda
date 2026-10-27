// check_schafer_turek.cpp — Schafer-Turek 2D-1 benchmark (steady flow past a cylinder, Re = 20)
//
// Build + run (from C:\Users\diego\lbm):
//   g++ -O3 -march=native -fopenmp tests/check_schafer_turek.cpp -o check_schafer_turek.exe
//   ./check_schafer_turek.exe [N] [tau]      N = cells per diameter (multiple of 10, default 20)
//                                            tau = relaxation time (default 0.65, as in PLAN.md)
//
// Geometry and reference values: see SchaferTurek in lbm_core.h. tau fixes nu = (tau - 1/2)/3, and
// Re = U N / nu = 20 then fixes the lattice inflow speed U = 20 nu / N (max 1.5 U: keep it <~ 0.1).
// Runs until C_D and C_L stop changing, then compares them with the reference.

#include "../src/lbm_core.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv) {
    int N = argc > 1 ? std::atoi(argv[1]) : 20;
    real tau = argc > 2 ? std::atof(argv[2]) : 0.65;
    if (N <= 0 || N % 10) { std::printf("N must be a positive multiple of 10\n"); return 2; }
    real nu = (tau - 0.5) / 3, U = 20 * nu / N;

    SchaferTurek st(N, U);
    std::printf("Schafer-Turek 2D-1  N=%d  grid %dx%d  tau=%.3f  U=%.4f (max %.4f, Ma %.2f)\n", N, st.L.nx,
                st.L.ny, tau, U, 1.5 * U, 1.5 * U * std::sqrt(3.0));

    // steady when C_D and C_L both move by < 1e-8 (relative to C_D) over CHECK steps; the flow
    // around the cylinder settles in a few D^2/nu, so cap at 100 of those
    const int CHECK = 1000;
    const long long maxSteps = (long long)(100.0 * N * N / nu);
    double cdPrev = 0, clPrev = 0, cd = 0, cl = 0;
    long long t;
    bool converged = false;
    auto t0 = std::chrono::steady_clock::now();
    for (t = 1; t <= maxSteps; t++) {
        st.L.step();
        if (t % CHECK) continue;
        Force F = st.force();
        cd = st.cd(F); cl = st.cl(F);
        if (std::isnan(cd)) { std::printf("FAIL: diverged (NaN) at step %lld\n", t); return 1; }
        if (t % (20 * CHECK) == 0) std::printf("  step %7lld  C_D = %.6f  C_L = %+.6f\n", t, cd, cl);
        if (std::fabs(cd - cdPrev) < 1e-8 * cd && std::fabs(cl - clPrev) < 1e-8 * cd) { converged = true; break; }
        cdPrev = cd; clPrev = cl;
    }
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    double eCd = (cd - SchaferTurek::CD_REF) / SchaferTurek::CD_REF;
    double eCl = (cl - SchaferTurek::CL_REF) / SchaferTurek::CL_REF;
    std::printf("\n(%lld steps, %.1f s, %.1f MLUPS)\n", t, secs, double(t) * st.L.nx * st.L.ny / secs / 1e6);
    std::printf("  C_D = %.5f   reference %.5f   error %+.2f%%\n", cd, SchaferTurek::CD_REF, 100 * eCd);
    std::printf("  C_L = %.5f   reference %.5f   error %+.1f%%\n", cl, SchaferTurek::CL_REF, 100 * eCl);

    // Staircase bounce-back makes the cylinder slightly the wrong shape and size, so C_D is off by an
    // amount that shrinks with N. Measured at tau = 0.65 when this test was written:
    //   N = 10: +17.1%   N = 20: +5.3%   N = 40: +1.5%
    // That error vs N IS the convergence study (research question 2), so this only guards against
    // regressions at N >= 20. C_L is tiny (1/500 of C_D, set by the 0.005 offset of the cylinder from
    // the channel middle) and more sensitive to the staircase: reported, not asserted.
    bool okCd = N < 20 || std::fabs(eCd) < 0.07;
    std::printf("\n[%s] reached steady state\n", converged ? "PASS" : "FAIL");
    std::printf("[%s] C_D within 7%% of the reference%s\n", okCd ? "PASS" : "FAIL", N < 20 ? " (not asserted below N = 20)" : "");
    return (converged && okCd) ? 0 : 1;
}
