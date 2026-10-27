// check_gpu.cu — the GPU solver must reproduce the CPU solver, then: how fast is it?
//
// Build + run (from C:\Users\diego\lbm, any terminal):
//   .\build_gpu.bat          (sets up the MSVC compiler nvcc needs, then builds this file)
//   .\check_gpu.exe
//
// Part 1 (PLAN.md, Tue 20): the same case on CPU and GPU in double for 1000 steps, then compare the
// post-collision populations (CPU: L.f2), rho/u and the wall force. They differ only by round-off
// (the GPU fuses multiply-adds and sums the force in a different order), so ~1e-13 or less.
// One case per code path: forced periodic channel, equilibrium inlet + cylinder, and
// Schafer-Turek with the velocity inlet / pressure outlet.
//
// Part 2 (PLAN.md, Thu 22): MLUPS on an empty periodic box, float and double, a few sizes.
// Effective bandwidth = MLUPS * (9 reads + 9 writes) * sizeof(T).

#include "../src/lbm_gpu.cuh"
#include <algorithm>
#include <chrono>

struct Diff { double f, u, force; };

Diff compare(const Lattice& L, const LatticeGPU<double>& g, Force Fc, Force Fg) {
    std::vector<real> post = g.postCollision();
    Lattice M = L;
    g.downloadMoments(M);
    Diff d = {0, 0, 0};
    for (int y = 0; y < L.ny; y++)
        for (int x = 0; x < L.nx; x++) {
            int n = L.idx(x, y);
            if (L.solid[n]) continue;
            for (int i = 0; i < 9; i++) d.f = std::max(d.f, std::fabs(post[L.fidx(i, x, y)] - L.f2[L.fidx(i, x, y)]));
            d.u = std::max(d.u, std::max(std::fabs(M.ux[n] - L.ux[n]), std::fabs(M.uy[n] - L.uy[n])));
            d.u = std::max(d.u, std::fabs(M.rho[n] - L.rho[n]));
        }
    double fmag = std::sqrt(Fc.x * Fc.x + Fc.y * Fc.y);
    d.force = std::sqrt((Fg.x - Fc.x) * (Fg.x - Fc.x) + (Fg.y - Fc.y) * (Fg.y - Fc.y)) / fmag;
    return d;
}

// run both for `steps` and compare; onBody picks the solid nodes the force is measured on
template <class Pred>
bool runCase(const char* name, Lattice& L, Pred onBody, int steps) {
    LatticeGPU<double> g(L);
    g.setForceBody(onBody);
    for (int t = 0; t < steps; t++) { L.step(); g.step(); }
    Force Fc = L.wallForce(onBody), Fg = g.wallForce();
    Diff d = compare(L, g, Fc, Fg);
    bool ok = d.f < 1e-12 && d.u < 1e-12 && d.force < 1e-10;
    std::printf("[%s] %-28s max |f| diff %.1e   |rho,u| diff %.1e   force rel. diff %.1e\n",
                ok ? "PASS" : "FAIL", name, d.f, d.u, d.force);
    return ok;
}

template <class T>
double gpuMLUPS(int n) {
    Lattice L(n, n);
    L.periodicX = true;
    L.setViscosity(0.1);
    L.initEquilibrium(1, 0.05, 0.02);
    LatticeGPU<T> g(L);
    g.setWriteMoments(false);
    int steps = std::max(50, int(4e9 / (double(n) * n)));  // ~a few seconds of work
    for (int t = 0; t < 10; t++) g.step();                 // warm-up
    g.sync();
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < steps; t++) g.step();
    g.sync();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return double(steps) * n * n / s / 1e6;
}

double cpuMLUPS(int n) {
    Lattice L(n, n);
    L.periodicX = true;
    L.setViscosity(0.1);
    L.initEquilibrium(1, 0.05, 0.02);
    int steps = std::max(10, int(2e8 / (double(n) * n)));
    auto t0 = std::chrono::steady_clock::now();
    for (int t = 0; t < steps; t++) L.step();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return double(steps) * n * n / s / 1e6;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (sm_%d%d, %.1f GB)\n\n", prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1e9);

    // ---- part 1: CPU vs GPU, double ----
    std::printf("CPU vs GPU, double, 1000 steps\n");
    const int STEPS = 1000;
    bool ok = true;
    {
        Lattice L(64, 34);  // channel H = 32, tau = 1
        L.setViscosity(0.5 / 3);
        setupChannel(L, 32, 0.1);
        ok &= runCase("channel (Guo force, periodic)", L, [](int, int) { return true; }, STEPS);
    }
    {
        const int NX = 400, NY = 100, CX = NX / 5, CY = NY / 2, R = NY / 9;
        Lattice L(NX, NY);
        L.setViscosity(0.1 * 2 * R / 100);
        setupFlowPast(L, 0.1, [&](int x, int y) { return (x - CX) * (x - CX) + (y - CY) * (y - CY) <= R * R; });
        kickWake(L, CX + R, CY, R, 0.01);
        ok &= runCase("cylinder (equilibrium inlet)", L, [](int, int) { return true; }, STEPS);
    }
    {
        SchaferTurek st(20, 0.05);
        ok &= runCase("Schafer-Turek (NEE in/out)", st.L, [&](int x, int y) { return st.inCylinder(x, y); }, STEPS);
    }

    // ---- part 2: performance ----
    std::printf("\nThroughput, empty periodic box (MLUPS = million lattice updates per second)\n");
    std::printf("      size     GPU float        GPU double       CPU double (all threads)\n");
    for (int n : {256, 512, 1024, 2048, 4096}) {
        double mf = gpuMLUPS<float>(n), md = gpuMLUPS<double>(n);
        std::printf("  %4d^2   %7.0f (%3.0f GB/s)  %7.0f (%3.0f GB/s)", n, mf, mf * 1e6 * 18 * 4 / 1e9, md,
                    md * 1e6 * 18 * 8 / 1e9);
        if (n <= 1024) std::printf("   %6.0f", cpuMLUPS(n));
        std::printf("\n");
    }

    std::printf("\n[%s] GPU matches CPU on all cases\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
