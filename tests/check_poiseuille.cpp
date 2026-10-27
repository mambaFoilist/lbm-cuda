// check_poiseuille.cpp — headless validation against the exact Poiseuille profile
//
// Build + run (from C:\Users\diego\lbm):
//   g++ -O3 -march=native -fopenmp tests/check_poiseuille.cpp -o check_poiseuille.exe
//   ./check_poiseuille.exe        (add -v to print the H=16 profile next to the exact one)
//
// Body-force-driven channel (see setupChannel in lbm_core.h), periodic in x, run to steady state.
// Exact answer: u(y) = F/(2 nu) (y - 0.5)(H + 0.5 - y), walls halfway between fluid and solid rows.
// Metric: relative L2 error  e = sqrt( sum (u_sim - u_exact)^2 / sum u_exact^2 )  over the fluid rows.
// The flow is uniform in x, so the lattice is a single column: 1 x (H + 2).
//
// Checks:
//   1. every run reaches steady state
//   2. at tau = 1 the error drops as 1/H^2 (bounce-back + Guo forcing are second order)
//   3. at the "magic" tau = 1/2 + sqrt(3/16) BGK + bounce-back is exact for Poiseuille, so the error
//      there is only round-off -> a sharp test of the forcing and walls. (In float32 this floor was
//      ~1e-4 at H = 32 and spoiled the order there; in double it is ~1e-13.)

#include "../src/lbm_core.h"
#include <chrono>
#include <cstdio>
#include <cstring>

const real UMAX = 0.1;  // centerline speed, same Mach number as the cylinder case

// forceErr: steady state means the walls hold back exactly what the body force pushes, so the
// momentum-exchange force on the walls must equal F_body * (number of fluid nodes) = fx * H here.
struct Result { double err, forceErr; long long steps; bool converged; };

Result runChannel(int H, real tau, bool printProfile = false) {
    Lattice L(1, H + 2);
    L.setViscosity((tau - 0.5) / 3);
    setupChannel(L, H, UMAX);
    const int CHECK = 500;
    // steady when the profile changes by < 1e-12 (relative) over CHECK steps. The slowest mode decays
    // like exp(-nu pi^2 t / H^2), so cap at ~50 of those time constants.
    const long long maxSteps = (long long)(50.0 * H * H / (L.nu * 9.87)) + 10 * CHECK;
    std::vector<double> prev(H + 1, 0.0);
    Result res = {0, 0, 0, false};
    for (res.steps = 1; res.steps <= maxSteps; res.steps++) {
        L.step();
        if (res.steps % CHECK) continue;
        double d = 0, n = 0;
        for (int y = 1; y <= H; y++) {
            double u = L.ux[L.idx(0, y)];
            d += (u - prev[y]) * (u - prev[y]);
            n += u * u;
            prev[y] = u;
        }
        if (std::isnan(n)) break;
        if (std::sqrt(d / n) < 1e-12) { res.converged = true; break; }
    }
    if (printProfile) {
        std::printf("     y    u_sim       u_exact\n");
        for (int y = 1; y <= H; y++) std::printf("  %4d  %.7f  %.7f\n", y, L.ux[L.idx(0, y)], poiseuilleU(L, H, y));
    }
    res.err = poiseuilleError(L, H, 0);
    Force Fw = L.wallForce();  // the loop always stops right after a step(), as wallForce needs
    double Fbody = L.fx * H;
    res.forceErr = std::sqrt((Fw.x - Fbody) * (Fw.x - Fbody) + Fw.y * Fw.y) / Fbody;
    return res;
}

int main(int argc, char** argv) {
    bool verbose = argc > 1 && std::strcmp(argv[1], "-v") == 0;
    auto t0 = std::chrono::steady_clock::now();
    bool okConv = true;
    double forceErrMax = 0;

    // ---- grid refinement at tau = 1 ----
    const int Hs[] = {4, 8, 16, 32, 64};
    const int NH = sizeof Hs / sizeof Hs[0];
    double err[NH], order[NH] = {};
    std::printf("Refinement, tau = 1.0, u_max = %.2f\n", UMAX);
    std::printf("     H    steps    L2 error   wall force vs body force    order\n");
    for (int k = 0; k < NH; k++) {
        Result r = runChannel(Hs[k], 1.0, verbose && Hs[k] == 16);
        err[k] = r.err;
        okConv &= r.converged;
        forceErrMax = std::fmax(forceErrMax, r.forceErr);
        std::printf("  %4d  %7lld   %.3e   %.1e", Hs[k], r.steps, r.err, r.forceErr);
        if (k > 0) {
            order[k] = std::log(err[k - 1] / err[k]) / std::log(double(Hs[k]) / Hs[k - 1]);
            std::printf("                    %.3f", order[k]);
        }
        std::printf("%s\n", r.converged ? "" : "  NOT CONVERGED");
    }
    double ordMin = 1e9, ordMax = -1e9;
    for (int k = 1; k < NH; k++) { ordMin = std::fmin(ordMin, order[k]); ordMax = std::fmax(ordMax, order[k]); }

    // ---- magic tau: discretization error vanishes, what's left is round-off ----
    const real TAU_MAGIC = 0.5 + std::sqrt(3.0 / 16.0);
    std::printf("\nMagic tau = %.4f (exact for BGK + bounce-back), error = round-off floor\n", TAU_MAGIC);
    Result m8 = runChannel(8, TAU_MAGIC), m32 = runChannel(32, TAU_MAGIC);
    okConv &= m8.converged && m32.converged;
    forceErrMax = std::fmax(forceErrMax, std::fmax(m8.forceErr, m32.forceErr));
    std::printf("  H =  8   L2 error %.3e\n  H = 32   L2 error %.3e\n", m8.err, m32.err);

    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    bool okOrder = ordMin > 1.95 && ordMax < 2.05;
    bool okErr = err[2] < 2e-3;                    // H = 16, tau = 1 (measured 1.78e-3)
    bool okMagic = m8.err < 1e-9 && m32.err < 1e-9; // a forcing or wall bug gives ~1e-2
    bool okForce = forceErrMax < 1e-9;              // momentum balance: wall force = body force

    std::printf("\n(%.1f s)\n", secs);
    std::printf("[%s] all runs reached steady state\n", okConv ? "PASS" : "FAIL");
    std::printf("[%s] convergence order in [%.3f, %.3f] over H = 4..64, expect 2\n", okOrder ? "PASS" : "FAIL", ordMin, ordMax);
    std::printf("[%s] L2 error at H=16 = %.3e (expect < 2e-3)\n", okErr ? "PASS" : "FAIL", err[2]);
    std::printf("[%s] magic-tau error at H=8, 32 = %.1e, %.1e (expect < 1e-9)\n", okMagic ? "PASS" : "FAIL", m8.err, m32.err);
    std::printf("[%s] wall force = body force, worst relative mismatch %.1e (expect < 1e-9)\n", okForce ? "PASS" : "FAIL", forceErrMax);
    return (okConv && okOrder && okErr && okMagic && okForce) ? 0 : 1;
}
