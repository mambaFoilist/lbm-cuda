// run_strouhal.cu — research question 1, part 2: does the wake shed at the right frequency?
//
// Build + run (from C:\Users\diego\lbm):
//   .\build_gpu.bat tests\run_strouhal.cu
//   .\run_strouhal.exe                       double, Re = 50 75 100 125 150   (~30-35 min total)
//   .\run_strouhal.exe float                 same in float                     (~6-8 min)
//   .\run_strouhal.exe double 100            just Re = 100 (pilot, ~5 min)
//   .\run_strouhal.exe double 100 --steps 20000    short smoke test
//
// Why this setup (vs check_cylinder.cpp, which gave St = 0.2018 at Re = 100, 23% high):
//   that domain was 400 x 100 with D = 22, i.e. the cylinder blocks 22% of the height, and with
//   periodic top/bottom it sits in a row of cylinders only 4.5 D apart. Confinement speeds up the
//   flow past the cylinder and raises both St and C_D. Here D = 40 in a 1600 x 800 box: blockage
//   D / ny = 5%, images 20 D apart, 10 D of inflow ahead, 30 D of wake behind.
//   U = 0.1 (Ma = 0.17), Re = U D / nu sets nu: tau = 3 U D / Re + 1/2 goes from 0.74 (Re 50) to
//   0.58 (Re 150). Equilibrium inlet + zero-gradient outlet, as in check_cylinder.
//
// Measurement: C_L = F_y / (1/2 U^2 D) every 10 steps (momentum exchange) -> runs/strouhal_Re<Re>_<T>.csv.
// The first half of each run is transient and is discarded; on the second half:
//   St = f D / U from upward zero crossings of C_L - mean (interpolated, timed over whole periods)
// scripts/make_figures.py also gets St from an FFT (Hann window + parabolic peak) of the same data.
// One summary row per run is appended to runs/strouhal.csv.
//
// Cost: 1.28 M cells; double ~690 MLUPS -> ~540 steps/s. 160k steps ~ 5 min, Re 50 (300k) ~ 9 min.
// Re = 50 is just above onset (Re_c ~ 47): the instability grows slowly, so it gets more steps.
//
// References (unconfined cylinder):
//   Williamson (1988, 1996), 49 < Re < 178:  St = -3.3265/Re + 0.1816 + 1.600e-4 Re
//   Roshko (1954), 50 < Re < 150:           St = 0.212 (1 - 21.2/Re)

#include "../src/lbm_gpu.cuh"
#include "run_util.h"
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

constexpr int NX = 1600, NY = 800, D = 40, R = D / 2, CXC = 400, CYC = NY / 2;
constexpr real U0 = 0.1;
constexpr int SAMPLE = 10;  // steps between force samples

double williamson(double Re) { return -3.3265 / Re + 0.1816 + 1.600e-4 * Re; }
double roshko(double Re) { return 0.212 * (1 - 21.2 / Re); }

struct Summary { double St, cdMean, clAmp, clRmsRatio; int periods; bool ok; };

// St and friends from the second half of the C_L / C_D record
Summary analyse(const std::vector<double>& t, const std::vector<double>& cd, const std::vector<double>& cl) {
    size_t n = t.size(), k0 = n / 2;
    Summary s = {0, 0, 0, 0, 0, false};
    if (n - k0 < 10) return s;
    double clMean = 0;
    for (size_t k = k0; k < n; k++) { clMean += cl[k]; s.cdMean += cd[k]; }
    clMean /= double(n - k0);
    s.cdMean /= double(n - k0);
    double tFirst = -1, tLast = -1, clMax = -1e9, clMin = 1e9;
    int crossings = 0;
    for (size_t k = k0 + 1; k < n; k++) {
        double a = cl[k - 1] - clMean, b = cl[k] - clMean;
        clMax = std::max(clMax, cl[k]); clMin = std::min(clMin, cl[k]);
        if (a < 0 && b >= 0) {
            double tc = t[k - 1] + (t[k] - t[k - 1]) * a / (a - b);
            if (tFirst < 0) tFirst = tc;
            tLast = tc;
            crossings++;
        }
    }
    s.clAmp = 0.5 * (clMax - clMin);
    s.periods = crossings - 1;
    if (s.periods >= 3) s.St = D / (U0 * (tLast - tFirst) / s.periods);
    // saturation check: rms of C_L in the 3rd quarter vs the 4th quarter of the run
    auto rms = [&](size_t a, size_t b) {
        double q = 0;
        for (size_t k = a; k < b; k++) q += (cl[k] - clMean) * (cl[k] - clMean);
        return std::sqrt(q / double(b - a));
    };
    s.clRmsRatio = rms(3 * n / 4, n) / rms(n / 2, 3 * n / 4);
    s.ok = s.periods >= 3 && std::fabs(s.clRmsRatio - 1) < 0.02;
    return s;
}

template <class T>
bool runRe(double Re, long long steps, const char* prec, FILE* summary) {
    Lattice L(NX, NY);
    L.setViscosity(U0 * D / Re);
    auto inCyl = [](int x, int y) { return (x - CXC) * (x - CXC) + (y - CYC) * (y - CYC) <= R * R; };
    setupFlowPast(L, U0, inCyl);
    kickWake(L, CXC + R, CYC, R, 0.1 * U0);  // break the symmetry so shedding starts (lbm_core.h)

    const double mlupsGuess = std::strcmp(prec, "float") == 0 ? 2000 : 650;
    std::printf("\nRe = %.0f   tau = %.4f   %s   %lld steps   (estimate %.1f min)\n", Re, L.tau, prec, steps,
                double(NX) * NY * steps / (mlupsGuess * 1e6) / 60);

    LatticeGPU<T> g(L);
    g.setWriteMoments(false);  // only forces are needed: skip writing rho/u (faster)
    g.setForceBody(inCyl);

    char name[96];
    std::snprintf(name, sizeof name, "runs/strouhal_Re%03.0f_%s.csv", Re, prec);
    char meta[256];
    std::snprintf(meta, sizeof meta, "Re=%.1f tau=%.6f U=%.3f D=%d nx=%d ny=%d blockage=%.3f cyl=(%d,%d) precision=%s steps=%lld",
                  Re, double(L.tau), double(U0), D, NX, NY, double(D) / NY, CXC, CYC, prec, steps);
    FILE* ts = openCsv(name, "step,C_D,C_L", meta);

    const double q = 0.5 * U0 * U0 * D;
    std::vector<double> tv, cdv, clv;
    tv.reserve(steps / SAMPLE + 1); cdv.reserve(steps / SAMPLE + 1); clv.reserve(steps / SAMPLE + 1);
    Stopwatch sw;
    for (long long t = 1; t <= steps; t++) {
        g.step();
        if (t % SAMPLE) continue;
        Force F = g.wallForce();  // right after step(), as it needs
        double cd = F.x / q, cl = F.y / q;
        if (std::isnan(cd)) { std::printf("  DIVERGED (NaN) at step %lld\n", t); std::fclose(ts); return false; }
        tv.push_back(double(t)); cdv.push_back(cd); clv.push_back(cl);
        std::fprintf(ts, "%lld,%.8e,%.8e\n", t, cd, cl);
        if (t % 20000 == 0)
            std::printf("  step %7lld  C_D = %.4f  C_L = %+.4f   (%.0f s, %.0f MLUPS)\n", t, cd, cl, sw.seconds(),
                        double(NX) * NY * t / sw.seconds() / 1e6);
    }
    std::fclose(ts);
    double secs = sw.seconds();

    Summary s = analyse(tv, cdv, clv);
    double stW = williamson(Re), stR = roshko(Re);
    std::printf("  St = %.4f (%d periods)   Williamson %.4f (%+.1f%%)   Roshko %.4f (%+.1f%%)\n", s.St, s.periods, stW,
                100 * (s.St / stW - 1), stR, 100 * (s.St / stR - 1));
    std::printf("  mean C_D = %.4f   C_L amplitude = %.4f   C_L rms 4th/3rd quarter = %.4f %s\n", s.cdMean, s.clAmp,
                s.clRmsRatio, s.ok ? "(saturated)" : "(NOT saturated: run longer)");
    std::printf("  -> %s  (%.0f s, %.0f MLUPS)\n", name, secs, double(NX) * NY * steps / secs / 1e6);

    std::fprintf(summary, "%.1f,%s,%.6f,%d,%lld,%.6f,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d,%.1f,%s\n", Re, prec, double(L.tau),
                 D, steps, s.St, s.periods, s.cdMean, s.clAmp, s.clRmsRatio, stW, stR, s.ok ? 1 : 0, secs,
                 gitHash().c_str());
    std::fflush(summary);
    return s.ok;
}

int main(int argc, char** argv) {
    std::string prec = "double";
    std::vector<double> Res;
    long long stepsOverride = 0;
    for (int a = 1; a < argc; a++) {
        if (!std::strcmp(argv[a], "float") || !std::strcmp(argv[a], "double")) prec = argv[a];
        else if (!std::strcmp(argv[a], "--steps") && a + 1 < argc) stepsOverride = std::atoll(argv[++a]);
        else Res.push_back(std::atof(argv[a]));
    }
    if (Res.empty()) Res = {50, 75, 100, 125, 150};

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s   domain %d x %d, D = %d (blockage %.0f%%), U = %.2f, %s\n", prop.name, NX, NY, D,
                100.0 * D / NY, double(U0), prec.c_str());

    FILE* summary = openCsv("runs/strouhal.csv",
                            "Re,precision,tau,D,steps,St_zc,periods,CD_mean,CL_amp,CL_rms_ratio,St_williamson,"
                            "St_roshko,saturated,seconds,git",
                            "", true);
    bool ok = true;
    for (double Re : Res) {
        long long steps = stepsOverride ? stepsOverride : (Re < 60 ? 300000 : 160000);
        ok &= prec == "float" ? runRe<float>(Re, steps, "float", summary) : runRe<double>(Re, steps, "double", summary);
    }
    std::fclose(summary);
    std::printf("\n[%s] every run reached a saturated shedding cycle\n", ok ? "PASS" : "WARN");
    return ok ? 0 : 1;
}
