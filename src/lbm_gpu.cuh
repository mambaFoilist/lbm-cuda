// lbm_gpu.cuh — CUDA port of the solver core (lbm_core.h): same physics, same boundary conditions.
//
// Set up a case on the CPU (a Lattice from lbm_core.h), then hand it over:
//     LatticeGPU<double> g(L);     // copies populations, walls, parameters
//     g.step();                    // as many times as L.step() would be called
//     g.downloadMoments(L);        // rho, ux, uy back into L (for output / plots)
//
// One fused kernel per step, one thread per node ("pull" scheme, PLAN.md ch. 9.3):
//   pull the 9 populations from the neighbours (bounce-back off walls)
//   -> inlet/outlet boundary conditions on boundary nodes -> BGK (+ Guo) collision -> write.
// Each population is read once and written once per step, which is what bandwidth-bound LBM wants.
// Two buffers (ping-pong) since a node's sources must not be overwritten while others still read them.
//
// The GPU buffer holds POST-collision populations; the CPU keeps the same thing in L.f2 right after
// L.step() (see wallForce in lbm_core.h). So after n steps on each, g.postCollision() and L.f2 should
// agree to round-off: tests/check_gpu.cu checks exactly that. The very first g.step() only collides
// (the starting populations are pre-collision, like the CPU's), every later one streams and collides.
//
// T = double for validation, float for the precision/performance comparison. Note FP64 on GeForce
// cards runs at 1/64 of FP32, so double may not be purely bandwidth-bound here.

#pragma once
#include "lbm_core.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t e_ = (call);                                                               \
        if (e_ != cudaSuccess) {                                                               \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

namespace gpu {

// D2Q9 tables as locals, so with the loops unrolled they fold into constants
#define D2Q9_TABLES                                              \
    constexpr int CX[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};        \
    constexpr int CY[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
#define D2Q9_OPP constexpr int OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

template <class T>
__device__ __forceinline__ T W(int i) { return i == 0 ? T(4) / T(9) : i < 5 ? T(1) / T(9) : T(1) / T(36); }

template <class T>
__device__ __forceinline__ T feq(int i, T r, T u, T v) {
    D2Q9_TABLES
    T cu = 3 * (CX[i] * u + CY[i] * v);
    return W<T>(i) * r * (1 + cu + T(0.5) * cu * cu - T(1.5) * (u * u + v * v));
}

template <class T>
struct Params {
    int nx, ny;
    T omega, fx, fy, uIn;
    bool periodicX, velocityInletPressureOutlet, forced, writeMoments;
    const uint8_t* solid;
    const T* uInY;  // null: uniform uIn
    T *rho, *ux, *uy;
};

__device__ __forceinline__ int fidx(int i, int x, int y, int nx, int ny) { return (i * ny + y) * nx + x; }

// populations arriving at (x, y) this step, as in the CPU streaming loop
template <class T>
__device__ void pull(const Params<T>& p, const T* fin, int x, int y, T f[9]) {
    D2Q9_TABLES
    D2Q9_OPP
#pragma unroll
    for (int i = 0; i < 9; i++) {
        int xs = x - CX[i];
        if (p.periodicX) xs = (xs + p.nx) % p.nx;
        int ys = (y - CY[i] + p.ny) % p.ny;
        if (xs < 0 || xs >= p.nx)               f[i] = fin[fidx(i, x, y, p.nx, p.ny)];  // fixed by the BCs
        else if (p.solid[ys * p.nx + xs])        f[i] = fin[fidx(OPP[i], x, y, p.nx, p.ny)];
        else                                     f[i] = fin[fidx(i, xs, ys, p.nx, p.ny)];
    }
}

// non-equilibrium extrapolation from the neighbour's just-streamed populations fn (see lbm_core.h)
template <class T>
__device__ void extrapolateNonEq(const T fn[9], bool pressure, T ub, T vb, T f[9]) {
    D2Q9_TABLES
    T r = 0, mx = 0, my = 0;
#pragma unroll
    for (int i = 0; i < 9; i++) { r += fn[i]; mx += fn[i] * CX[i]; my += fn[i] * CY[i]; }
    T un = mx / r, vn = my / r, rb = pressure ? T(1) : r;
    if (pressure) { ub = un; vb = vn; }
#pragma unroll
    for (int i = 0; i < 9; i++) f[i] = feq<T>(i, rb, ub, vb) + fn[i] - feq<T>(i, r, un, vn);
}

template <class T>
__global__ void streamCollide(Params<T> p, const T* __restrict__ fin, T* __restrict__ fout, bool collideOnly) {
    D2Q9_TABLES
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= p.nx * p.ny || p.solid[n]) return;
    int x = n % p.nx, y = n / p.nx;

    T f[9];
    if (collideOnly) {
#pragma unroll
        for (int i = 0; i < 9; i++) f[i] = fin[fidx(i, x, y, p.nx, p.ny)];
    } else if (p.periodicX || (x != 0 && x != p.nx - 1)) {
        pull(p, fin, x, y, f);
    } else if (p.velocityInletPressureOutlet) {
        if (p.solid[y * p.nx] || p.solid[y * p.nx + p.nx - 1]) {
            pull(p, fin, x, y, f);  // the CPU skips the BC on rows with a solid end
        } else {
            T fn[9];
            pull(p, fin, x == 0 ? 1 : p.nx - 2, y, fn);
            T ui = p.uInY ? p.uInY[y] : p.uIn;
            if (x == 0) extrapolateNonEq<T>(fn, false, ui, T(0), f);  // velocity inlet
            else        extrapolateNonEq<T>(fn, true, T(0), T(0), f); // pressure outlet, rho = 1
        }
    } else if (x == 0) {  // equilibrium inlet
        T ui = p.uInY ? p.uInY[y] : p.uIn;
#pragma unroll
        for (int i = 0; i < 9; i++) f[i] = feq<T>(i, T(1), ui, T(0));
    } else {  // zero-gradient outlet: what arrived at the node before it
        pull(p, fin, p.nx - 2, y, f);
    }

    // BGK collision, with Guo forcing (same as Lattice::step)
    T r = 0, mx = 0, my = 0;
#pragma unroll
    for (int i = 0; i < 9; i++) { r += f[i]; mx += f[i] * CX[i]; my += f[i] * CY[i]; }
    T u = mx / r, v = my / r;
    if (p.forced) { u += T(0.5) * p.fx / r; v += T(0.5) * p.fy / r; }
    if (p.writeMoments) { p.rho[n] = r; p.ux[n] = u; p.uy[n] = v; }
#pragma unroll
    for (int i = 0; i < 9; i++) {
        T fi = f[i] + p.omega * (feq<T>(i, r, u, v) - f[i]);
        if (p.forced) {
            T cF = CX[i] * p.fx + CY[i] * p.fy, cu = CX[i] * u + CY[i] * v;
            fi += (1 - T(0.5) * p.omega) * W<T>(i) * (3 * (cF - u * p.fx - v * p.fy) + 9 * cu * cF);
        }
        fout[fidx(i, x, y, p.nx, p.ny)] = fi;
    }
}

// momentum exchange on the solid nodes flagged in body (see Lattice::wallForce); out = {Fx, Fy}
template <class T>
__global__ void wallForceKernel(Params<T> p, const T* __restrict__ f, const uint8_t* __restrict__ body, double* out) {
    D2Q9_TABLES
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= p.nx * p.ny || p.solid[n]) return;
    int x = n % p.nx, y = n / p.nx;
    double sx = 0, sy = 0;
#pragma unroll
    for (int i = 1; i < 9; i++) {
        int xn = x + CX[i];
        if (p.periodicX) xn = (xn + p.nx) % p.nx;
        else if (xn < 0 || xn >= p.nx) continue;
        int m = ((y + CY[i] + p.ny) % p.ny) * p.nx + xn;
        if (!p.solid[m] || !body[m]) continue;
        double fi = f[fidx(i, x, y, p.nx, p.ny)];
        sx += 2 * fi * CX[i];
        sy += 2 * fi * CY[i];
    }
    if (sx != 0 || sy != 0) { atomicAdd(out, sx); atomicAdd(out + 1, sy); }
}

}  // namespace gpu

template <class T>
struct LatticeGPU {
    int nx, ny;
    gpu::Params<T> p{};
    T *fA = nullptr, *fB = nullptr;  // fA = current state (post-collision once primed), fB = scratch
    T *dRho = nullptr, *dUx = nullptr, *dUy = nullptr, *dUInY = nullptr;
    uint8_t *dSolid = nullptr, *dBody = nullptr;
    double* dForce = nullptr;
    bool primed = false;              // false until the first (collide-only) step
    static constexpr int BLOCK = 256;

    explicit LatticeGPU(const Lattice& L) : nx(L.nx), ny(L.ny) {
        size_t N = size_t(nx) * ny;
        CUDA_CHECK(cudaMalloc(&fA, 9 * N * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&fB, 9 * N * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&dRho, N * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&dUx, N * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&dUy, N * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&dSolid, N));
        CUDA_CHECK(cudaMalloc(&dBody, N));
        CUDA_CHECK(cudaMalloc(&dForce, 2 * sizeof(double)));
        upload(dSolid, L.solid);
        upload(dBody, L.solid);  // force on every solid node until setForceBody says otherwise
        upload(fA, L.f);         // pre-collision start; the first step only collides it
        if (!L.uInY.empty()) {
            CUDA_CHECK(cudaMalloc(&dUInY, ny * sizeof(T)));
            upload(dUInY, L.uInY);
        }
        p.nx = nx; p.ny = ny;
        p.omega = T(L.omega); p.fx = T(L.fx); p.fy = T(L.fy); p.uIn = T(L.uIn);
        p.periodicX = L.periodicX;
        p.velocityInletPressureOutlet = L.velocityInletPressureOutlet;
        p.forced = L.fx != 0 || L.fy != 0;
        p.writeMoments = true;
        p.solid = dSolid; p.uInY = dUInY;
        p.rho = dRho; p.ux = dUx; p.uy = dUy;
    }
    LatticeGPU(const LatticeGPU&) = delete;
    LatticeGPU& operator=(const LatticeGPU&) = delete;
    ~LatticeGPU() {
        for (void* q : {(void*)fA, (void*)fB, (void*)dRho, (void*)dUx, (void*)dUy, (void*)dUInY,
                        (void*)dSolid, (void*)dBody, (void*)dForce})
            cudaFree(q);
    }

    // skip writing rho/u every step (benchmarks): saves 3 of the 21 values moved per node
    void setWriteMoments(bool on) { p.writeMoments = on; }

    // change the viscosity mid-run (e.g. the viewer's Re keys); tau = 3 nu + 1/2 must stay > 0.5
    void setViscosity(real nu) { p.omega = T(1 / (3 * nu + real(0.5))); }

    void step() {
        int blocks = int((size_t(nx) * ny + BLOCK - 1) / BLOCK);
        gpu::streamCollide<T><<<blocks, BLOCK>>>(p, fA, fB, !primed);  // read current, write the other
        CUDA_CHECK(cudaGetLastError());
        primed = true;
        std::swap(fA, fB);  // what was just written is now the current state
    }

    // which solid nodes wallForce() counts (e.g. the cylinder and not the channel walls)
    template <class Pred>
    void setForceBody(Pred onBody) {
        std::vector<uint8_t> m(size_t(nx) * ny);
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++) m[size_t(y) * nx + x] = onBody(x, y);
        upload(dBody, m);
    }

    Force wallForce() {
        CUDA_CHECK(cudaMemset(dForce, 0, 2 * sizeof(double)));
        int blocks = int((size_t(nx) * ny + BLOCK - 1) / BLOCK);
        gpu::wallForceKernel<T><<<blocks, BLOCK>>>(p, fA, dBody, dForce);
        CUDA_CHECK(cudaGetLastError());
        double h[2];
        CUDA_CHECK(cudaMemcpy(h, dForce, sizeof h, cudaMemcpyDeviceToHost));
        return {real(h[0]), real(h[1])};
    }

    // rho, ux, uy from the last step into L (needs setWriteMoments(true), the default)
    void downloadMoments(Lattice& L) const {
        download(L.rho, dRho); download(L.ux, dUx); download(L.uy, dUy);
    }

    // current post-collision populations, same layout as Lattice::f2
    std::vector<real> postCollision() const {
        std::vector<real> out(size_t(9) * nx * ny);
        download(out, fA);
        return out;
    }

    void sync() const { CUDA_CHECK(cudaDeviceSynchronize()); }

private:
    template <class D, class S>
    static void upload(D* dst, const std::vector<S>& src) {  // converts, e.g. double -> float
        std::vector<D> tmp(src.size());
        for (size_t k = 0; k < src.size(); k++) tmp[k] = static_cast<D>(src[k]);
        CUDA_CHECK(cudaMemcpy(dst, tmp.data(), tmp.size() * sizeof(D), cudaMemcpyHostToDevice));
    }
    template <class S>
    static void download(std::vector<real>& dst, const S* src) {
        std::vector<S> tmp(dst.size());
        CUDA_CHECK(cudaMemcpy(tmp.data(), src, tmp.size() * sizeof(S), cudaMemcpyDeviceToHost));
        dst.assign(tmp.begin(), tmp.end());
    }
};
