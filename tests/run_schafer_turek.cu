// run_schafer_turek.cu — research question 2: what does staircase bounce-back cost in accuracy?
//
// Build + run (from C:\Users\diego\lbm):
//   .\build_gpu.bat tests\run_schafer_turek.cu
//   .\run_schafer_turek.exe                     N = 10 20 40 80, 3 centre offsets each (~1 h total)
//   .\run_schafer_turek.exe 10 20 40            skip N = 80 (~5 min)
//   .\run_schafer_turek.exe 80 --offsets 2      only N = 80, 2 offsets (~35 min)
//
// Schafer-Turek 2D-1 (Re = 20, steady), the same case as check_schafer_turek.cpp (SchaferTurek in
// lbm_core.h), on the GPU in double, at N = cells per diameter.
//
// Why tau is held FIXED at 0.65 while N changes (PLAN.md §4):
//   Re = U N / nu with nu = (tau - 1/2)/3. Fixing tau fixes nu, so U = 20 nu / N = 1/N falls as the
//   grid is refined ("diffusive scaling"). Then
//     * the BGK wall-location error, which depends on tau (see sweep_poiseuille_tau), is the same
//       at every N instead of changing with it and muddling the trend;
//     * the compressibility error ~ Ma^2 ~ U^2 ~ 1/N^2 also falls at second order.
//   Catch: at N = 10 the peak inflow is 1.5 U = 0.15 (Ma 0.26), above the usual u <= 0.1. That is part
//   of the N = 10 error and it falls away quickly; N >= 20 is inside the limit.
//
// Why several sub-cell centre offsets (ox, oy) per N:
//   the staircase cylinder depends on where the circle lands relative to the grid, so one placement
//   gives one sample of a scattered error. Shifting the centre by a fraction of a cell resamples it;
//   the spread is the error bar. oy is kept within +-0.25 cells so the cylinder's height, which
//   sets C_L, moves by at most 0.25/N D (C_D barely depends on it). Report C_L spreads with that in mind.
//
// Steady state: C_D and C_L each change by < 1e-7 C_D over CHECK = max(1000, 2.5 N^2) steps.
// Relaxation times scale like N^2 / nu here, so scaling CHECK with N^2 makes the criterion equally
// strict at every N. N = 20 (CPU) needed ~100k steps -> expect ~250 N^2 steps.
//
// Cost (double ~690 MLUPS on big grids, less on small ones): N = 80 is 1760 x 330 = 0.58 M cells,
// ~1.6 M steps -> ~20 min per offset. N = 40 ~1.5 min, N <= 20 seconds.
//
// Output: one row per run appended to runs/schafer_turek.csv (rerunning a case adds a new row;
// make_figures.py keeps the last one per (N, ox, oy)).

#include "../src/lbm_gpu.cuh"
#include "run_util.h"
#include <algorithm>
#include <cstring>
#include <vector>

constexpr real TAU = 0.65, RE = 20;
const real OFFSETS[][2] = {{0.0, 0.0}, {0.3, 0.2}, {0.6, -0.2}, {0.15, -0.1}};
constexpr int MAX_OFFSETS = 4;

int main(int argc, char** argv) {
    std::vector<int> Ns;
    int nOff = 3;
    for (int a = 1; a < argc; a++) {
        if (!std::strcmp(argv[a], "--offsets") && a + 1 < argc) nOff = std::atoi(argv[++a]);
        else Ns.push_back(std::atoi(argv[a]));
    }
    if (Ns.empty()) Ns = {10, 20, 40, 80};
    nOff = std::max(1, std::min(nOff, MAX_OFFSETS));
    for (int N : Ns)
        if (N <= 0 || N % 10) { std::printf("N must be a positive multiple of 10 (got %d)\n", N); return 2; }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const real nu = (TAU - 0.5) / 3;
    std::printf("GPU: %s   Schafer-Turek 2D-1, tau = %.3f fixed, double, %d offset(s) per N\n", prop.name, double(TAU), nOff);
    double totalEst = 0;
    for (int N : Ns) totalEst += nOff * 22.0 * N * (41 * N / 10 + 2) * 250.0 * N * N / (N >= 40 ? 650e6 : 350e6);
    std::printf("Estimated total: %.0f min (cells x ~250 N^2 steps / MLUPS)\n", totalEst / 60);

    FILE* out = openCsv("runs/schafer_turek.csv",
                        "N,ox,oy,tau,U,nx,ny,steps,converged,C_D,C_L,err_CD,err_CL,seconds,MLUPS,git", "", true);
    bool allOk = true;
    for (int N : Ns)
        for (int k = 0; k < nOff; k++) {
            real ox = OFFSETS[k][0], oy = OFFSETS[k][1], U = RE * nu / N;
            SchaferTurek st(N, U, RE, ox, oy);
            LatticeGPU<double> g(st.L);
            g.setWriteMoments(false);
            g.setForceBody([&](int x, int y) { return st.inCylinder(x, y); });

            const long long CHECK = std::max(1000LL, (long long)(2.5 * N * N));
            const long long maxSteps = 1000LL * N * N;
            const double cells = double(st.L.nx) * st.L.ny;
            std::printf("\nN = %d  offset (%.2f, %.2f)  grid %d x %d  U = %.4f (max %.4f)  check every %lld\n", N,
                        double(ox), double(oy), st.L.nx, st.L.ny, double(U), 1.5 * double(U), CHECK);

            double cd = 0, cl = 0, cdPrev = 0, clPrev = 0;
            bool converged = false, nan = false;
            long long t;
            Stopwatch sw;
            for (t = 1; t <= maxSteps; t++) {
                g.step();
                if (t % CHECK) continue;
                Force F = g.wallForce();
                cd = st.cd(F); cl = st.cl(F);
                if (std::isnan(cd)) { nan = true; break; }
                if (t % (20 * CHECK) == 0)
                    std::printf("  step %8lld  C_D = %.6f  C_L = %+.6f   (%.0f s)\n", t, cd, cl, sw.seconds());
                if (std::fabs(cd - cdPrev) < 1e-7 * cd && std::fabs(cl - clPrev) < 1e-7 * cd) { converged = true; break; }
                cdPrev = cd; clPrev = cl;
            }
            double secs = sw.seconds();
            double eCd = (cd - SchaferTurek::CD_REF) / SchaferTurek::CD_REF;
            double eCl = (cl - SchaferTurek::CL_REF) / SchaferTurek::CL_REF;
            std::printf("  %s after %lld steps (%.0f s, %.0f MLUPS):  C_D = %.6f (%+.3f%%)  C_L = %+.6f (%+.1f%%)\n",
                        nan ? "DIVERGED" : converged ? "steady" : "NOT steady", t, secs, cells * t / secs / 1e6, cd,
                        100 * eCd, cl, 100 * eCl);
            std::fprintf(out, "%d,%.3f,%.3f,%.4f,%.6f,%d,%d,%lld,%d,%.10f,%.10f,%.6e,%.6e,%.1f,%.0f,%s\n", N, double(ox),
                         double(oy), double(TAU), double(U), st.L.nx, st.L.ny, t, converged ? 1 : 0, cd, cl, eCd, eCl,
                         secs, cells * t / secs / 1e6, gitHash().c_str());
            std::fflush(out);
            allOk &= converged;
        }
    std::fclose(out);
    std::printf("\n[%s] every run reached steady state   -> runs/schafer_turek.csv\n", allOk ? "PASS" : "FAIL");
    return allOk ? 0 : 1;
}
