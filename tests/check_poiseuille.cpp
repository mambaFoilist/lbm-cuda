// check_poiseuille.cpp — headless validation against the exact Poiseuille profile
//
// Build + run (from C:\Users\diego\lbm):
//   g++ -O3 -march=native -fopenmp tests/check_poiseuille.cpp -o check_poiseuille.exe -lgdi32
//   ./check_poiseuille.exe        (add -v to print the H=16 profile next to the exact one)
//
// Body-force-driven channel (see setupChannel in the solver), periodic in x, run to steady state.
// Exact answer: u(y) = F/(2 nu) (y - 0.5)(H + 0.5 - y), walls halfway between fluid and solid rows.
// Metric: relative L2 error  e = sqrt( sum (u_sim - u_exact)^2 / sum u_exact^2 )  over the fluid rows.
//
// Checks:
//   1. every run reaches steady state
//   2. at tau = 1 the error drops as 1/H^2 (bounce-back + Guo forcing are second order)
//   3. at the "magic" tau = 1/2 + sqrt(3/16) BGK + bounce-back is exact for Poiseuille, so the error
//      there is only float32 round-off -> a sharp test of the forcing and walls, and a measure of the
//      precision floor. The floor grows ~H^2 (the per-step change drops below one float ulp and the
//      populations freeze early), which is why the order is only asserted on H = 4, 8, 16.

#define NOMINMAX
#define main viewer_main
#include "../src/lbm_viewer.cpp"
#undef main
#include <chrono>
#include <cstring>

const float UMAX = 0.1f;  // centerline speed, same Mach number as the cylinder case

struct Result { double err; long long steps; bool converged; };

Result runChannel(int H, float tau, bool printProfile = false) {
    setViscosity((tau - 0.5f) / 3.0f);
    setupChannel(H, UMAX);
    const int X = NX / 2, CHECK = 500;
    // steady when the profile changes by < 1e-7 (relative) over CHECK steps. The slowest mode decays
    // like exp(-nu pi^2 t / H^2), so cap at ~50 of those time constants.
    const long long maxSteps = (long long)(50.0 * H * H / (NU * 9.87)) + 10 * CHECK;
    std::vector<double> prev(H + 1, 0.0);
    Result res = {0, 0, false};
    for (res.steps = 1; res.steps <= maxSteps; res.steps++) {
        step();
        if (res.steps % CHECK) continue;
        double d = 0, n = 0;
        for (int y = 1; y <= H; y++) {
            double u = ux[idx(X, y)];
            d += (u - prev[y]) * (u - prev[y]);
            n += u * u;
            prev[y] = u;
        }
        if (std::isnan(n)) break;
        if (std::sqrt(d / n) < 1e-7) { res.converged = true; break; }
    }
    if (printProfile) {
        std::printf("     y    u_sim       u_exact\n");
        for (int y = 1; y <= H; y++) std::printf("  %4d  %.7f  %.7f\n", y, ux[idx(X, y)], channelExactU(y));
    }
    res.err = channelProfileError(X);
    return res;
}

int main(int argc, char** argv) {
    bool verbose = argc > 1 && std::strcmp(argv[1], "-v") == 0;
    auto t0 = std::chrono::steady_clock::now();
    bool okConv = true;

    // ---- grid refinement at tau = 1 ----
    const int Hs[] = {4, 8, 16, 32};
    const int NH = sizeof Hs / sizeof Hs[0];
    double err[NH];
    std::printf("Refinement, tau = 1.0, u_max = %.2f\n", UMAX);
    std::printf("     H    steps    L2 error    order\n");
    for (int k = 0; k < NH; k++) {
        Result r = runChannel(Hs[k], 1.0f, verbose && Hs[k] == 16);
        err[k] = r.err;
        okConv &= r.converged;
        std::printf("  %4d  %7lld   %.3e", Hs[k], r.steps, r.err);
        if (k > 0) std::printf("    %.2f%s", std::log(err[k - 1] / err[k]) / std::log(double(Hs[k]) / Hs[k - 1]),
                               Hs[k] > 16 ? "  (float32 floor, not asserted)" : "");
        std::printf("%s\n", r.converged ? "" : "  NOT CONVERGED");
    }
    double p48 = std::log(err[0] / err[1]) / std::log(2.0), p816 = std::log(err[1] / err[2]) / std::log(2.0);

    // ---- magic tau: discretization error vanishes, what's left is round-off ----
    const float TAU_MAGIC = 0.5f + std::sqrt(3.0f / 16.0f);
    std::printf("\nMagic tau = %.4f (exact for BGK + bounce-back), error = float32 floor\n", TAU_MAGIC);
    Result m8 = runChannel(8, TAU_MAGIC), m32 = runChannel(32, TAU_MAGIC);
    okConv &= m8.converged && m32.converged;
    std::printf("  H =  8   L2 error %.3e\n  H = 32   L2 error %.3e\n", m8.err, m32.err);

    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    bool okOrder = p48 > 1.9 && p48 < 2.1 && p816 > 1.9 && p816 < 2.1;
    bool okErr = err[2] < 2e-3;   // H = 16, tau = 1 (measured 1.76e-3 when this test was written)
    bool okMagic = m8.err < 1e-4; // measured ~6e-6; a forcing or wall bug gives ~1e-2

    std::printf("\n(%.1f s)\n", secs);
    std::printf("[%s] all runs reached steady state\n", okConv ? "PASS" : "FAIL");
    std::printf("[%s] convergence order %.2f (H 4->8), %.2f (H 8->16), expect 2\n", okOrder ? "PASS" : "FAIL", p48, p816);
    std::printf("[%s] L2 error at H=16 = %.3e (expect < 2e-3)\n", okErr ? "PASS" : "FAIL", err[2]);
    std::printf("[%s] magic-tau error at H=8 = %.3e (expect < 1e-4)\n", okMagic ? "PASS" : "FAIL", m8.err);
    return (okConv && okOrder && okErr && okMagic) ? 0 : 1;
}
