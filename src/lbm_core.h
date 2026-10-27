// lbm_core.h — D2Q9 lattice Boltzmann (BGK) solver core. No window code: the viewer, the tests and
// (later) the GPU version all build on this.
//
// Units are lattice units: dx = dt = 1, c_s^2 = 1/3.
// Everything below maps to 4 equations:
//   1. moments:      rho = sum f_i,   rho*u = sum f_i * c_i
//   2. equilibrium:  f_i^eq = w_i rho (1 + 3 c.u + 4.5 (c.u)^2 - 1.5 u.u)
//   3. collide:      f_i <- f_i - (f_i - f_i^eq) / tau
//   4. stream:       f_i(x + c_i, t+1) = f_i(x, t)
// Chapman-Enskog shows this recovers Navier-Stokes with nu = c_s^2 (tau - 1/2), p = c_s^2 rho.

#pragma once
#include <cmath>
#include <cstdint>
#include <vector>

// Double for validation: in float the populations stop resolving the per-step change once the
// flow is slow or the grid is fine (Poiseuille H = 32 already sat on the float floor).
using real = double;

// ---------------- D2Q9 lattice ----------------
//   6 2 5
//   3 0 1
//   7 4 8
inline constexpr int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
inline constexpr int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
// written as real(1)/9, not 1.f/9: float literals would make the weights sum to 1 +- 1e-8 and leak mass
inline constexpr real w[9] = {real(4) / 9, real(1) / 9, real(1) / 9, real(1) / 9, real(1) / 9,
                              real(1) / 36, real(1) / 36, real(1) / 36, real(1) / 36};
inline constexpr int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};  // direction pointing the other way

inline real feq(int i, real r, real u, real v) {
    real cu = 3 * (cx[i] * u + cy[i] * v);
    return w[i] * r * (1 + cu + real(0.5) * cu * cu - real(1.5) * (u * u + v * v));
}

// ---------------- lattice ----------------
struct Force { real x, y; };

struct Lattice {
    int nx, ny;
    std::vector<real> f, f2;          // populations (two buffers for streaming)
    std::vector<real> rho, ux, uy;    // moments, written by the collision
    std::vector<uint8_t> solid;

    real nu = 0, tau = 0.5, omega = 2;
    real fx = 0, fy = 0;      // body force density, added with Guo's scheme in the collision
    bool periodicX = false;   // left/right wrap around; otherwise inlet at x = 0, outlet at x = nx-1
    real uIn = 0;             // inlet speed (when !periodicX)

    Lattice(int nx, int ny)
        : nx(nx), ny(ny), f(9 * nx * ny), f2(9 * nx * ny), rho(nx * ny, 1), ux(nx * ny), uy(nx * ny),
          solid(nx * ny) {}

    int idx(int x, int y) const { return y * nx + x; }
    int fidx(int i, int x, int y) const { return (i * ny + y) * nx + x; }

    void setViscosity(real nu_) { nu = nu_; tau = 3 * nu + real(0.5); omega = 1 / tau; }  // tau must stay > 0.5

    // every node to equilibrium at (r, u, v); solid nodes at rest
    void initEquilibrium(real r, real u, real v) {
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++) {
                int n = idx(x, y);
                real un = solid[n] ? 0 : u, vn = solid[n] ? 0 : v;
                rho[n] = r; ux[n] = un; uy[n] = vn;
                for (int i = 0; i < 9; i++) f[fidx(i, x, y)] = feq(i, r, un, vn);
            }
    }

    void step() {
        const bool forced = fx != 0 || fy != 0;  // unforced path stays exactly the plain BGK update

        // 1-3: compute moments, then BGK collision (in place)
        //   With a force (Guo et al. 2002): u = (sum f_i c_i + F/2) / rho, and after relaxing add
        //   S_i = (1 - omega/2) w_i [3 (c_i - u).F + 9 (c_i.u)(c_i.F)]. Keeps the scheme second order.
#pragma omp parallel for if (nx * ny > 4096)  // tiny grids: thread startup costs more than the work
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++) {
                if (solid[idx(x, y)]) continue;
                real r = 0, mx = 0, my = 0;
                for (int i = 0; i < 9; i++) {
                    real fi = f[fidx(i, x, y)];
                    r += fi;
                    mx += fi * cx[i];
                    my += fi * cy[i];
                }
                real u = mx / r, v = my / r;
                if (forced) { u += real(0.5) * fx / r; v += real(0.5) * fy / r; }
                rho[idx(x, y)] = r; ux[idx(x, y)] = u; uy[idx(x, y)] = v;
                for (int i = 0; i < 9; i++) {
                    real& fi = f[fidx(i, x, y)];
                    fi += omega * (feq(i, r, u, v) - fi);
                    if (forced) {
                        real cF = cx[i] * fx + cy[i] * fy, cu = cx[i] * u + cy[i] * v;
                        fi += (1 - real(0.5) * omega) * w[i] * (3 * (cF - u * fx - v * fy) + 9 * cu * cF);
                    }
                }
            }

        // 4: streaming ("pull": each node grabs what arrives from x - c_i).
        //    If the source is a wall, the population that left toward the wall
        //    bounces back: f_i(x) = f_opp(i)(x)  -> no-slip wall (halfway bounce-back).
        //    Top/bottom are periodic; left/right too if periodicX.
#pragma omp parallel for if (nx * ny > 4096)  // tiny grids: thread startup costs more than the work
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++) {
                if (solid[idx(x, y)]) continue;
                for (int i = 0; i < 9; i++) {
                    int xs = x - cx[i];
                    if (periodicX) xs = (xs + nx) % nx;
                    int ys = (y - cy[i] + ny) % ny;
                    if (xs < 0 || xs >= nx)        f2[fidx(i, x, y)] = f[fidx(i, x, y)];  // fixed by BCs below
                    else if (solid[idx(xs, ys)])   f2[fidx(i, x, y)] = f[fidx(opp[i], x, y)];
                    else                           f2[fidx(i, x, y)] = f[fidx(i, xs, ys)];
                }
            }

        // boundaries: inlet = equilibrium at (rho=1, u=uIn); outlet = zero-gradient copy
        if (!periodicX)
            for (int y = 0; y < ny; y++)
                for (int i = 0; i < 9; i++) {
                    f2[fidx(i, 0, y)] = feq(i, 1, uIn, 0);
                    f2[fidx(i, nx - 1, y)] = f2[fidx(i, nx - 2, y)];
                }

        f.swap(f2);
    }

    // Force of the fluid on the solid nodes picked by onBody(x, y), by momentum exchange (Ladd 1994).
    // A population f*_i (post-collision) at fluid node x heading into solid at x + c_i comes back as
    // f_opp(i): its momentum flips from f*_i c_i to -f*_i c_i, so the wall receives 2 f*_i c_i.
    // F = sum over all such links of 2 f*_i c_i  (stationary walls).
    // Call right after step(): the swap leaves that step's post-collision populations in f2.
    template <class Pred>
    Force wallForce(Pred onBody) const {
        real sx = 0, sy = 0;
#pragma omp parallel for reduction(+ : sx, sy) if (nx * ny > 4096)
        for (int y = 0; y < ny; y++)
            for (int x = 0; x < nx; x++) {
                if (solid[idx(x, y)]) continue;
                for (int i = 1; i < 9; i++) {
                    int xn = x + cx[i];
                    if (periodicX) xn = (xn + nx) % nx;
                    else if (xn < 0 || xn >= nx) continue;
                    int yn = (y + cy[i] + ny) % ny;
                    if (!solid[idx(xn, yn)] || !onBody(xn, yn)) continue;
                    real fi = f2[fidx(i, x, y)];
                    sx += 2 * fi * cx[i];
                    sy += 2 * fi * cy[i];
                }
            }
        return {sx, sy};
    }
    Force wallForce() const { return wallForce([](int, int) { return true; }); }  // on every solid node
};

// ---------------- cases ----------------

// Uniform flow at u0 from an equilibrium inlet past whatever inSolid(x, y) marks as solid
// (a cylinder, a plate, ...). Top/bottom periodic, zero-gradient outlet.
template <class Pred>
void setupFlowPast(Lattice& L, real u0, Pred inSolid) {
    L.periodicX = false;
    L.fx = L.fy = 0;
    L.uIn = u0;
    for (int y = 0; y < L.ny; y++)
        for (int x = 0; x < L.nx; x++) L.solid[L.idx(x, y)] = inSolid(x, y);
    L.initEquilibrium(1, u0, 0);
}

// A symmetric body in a symmetric flow stays symmetric until round-off breaks it: in double that takes
// tens of thousands of steps. Offsetting the body doesn't help when top/bottom are periodic (every
// height is equivalent). Instead, give the fluid in a size x size box a small sideways velocity v at
// the start; it washes out downstream and leaves the wake shedding.
inline void kickWake(Lattice& L, int x0, int y0, int size, real v) {
    for (int y = y0; y < y0 + size; y++)
        for (int x = x0; x < x0 + size; x++) {
            int n = L.idx(x, (y + L.ny) % L.ny);
            if (L.solid[n]) continue;
            L.uy[n] = v;
            for (int i = 0; i < 9; i++) L.f[L.fidx(i, x, (y + L.ny) % L.ny)] = feq(i, L.rho[n], L.ux[n], v);
        }
}

// Channel for the Poiseuille check: fluid rows y = 1..H, solid at y = 0 and y > H (needs ny >= H + 2),
// periodic in x, driven by a uniform body force. With halfway bounce-back the walls sit at y = 0.5 and
// y = H + 0.5, and the exact steady profile is u(y) = F / (2 nu) * (y - 0.5) * (H + 0.5 - y)  (rho = 1).
// F is chosen so the centerline speed is umax. Call setViscosity first: F depends on nu.
inline void setupChannel(Lattice& L, int H, real umax) {
    L.periodicX = true;
    L.fx = 8 * L.nu * umax / (real(H) * H);  // u_max = F H^2 / (8 nu)
    L.fy = 0;
    for (int y = 0; y < L.ny; y++)
        for (int x = 0; x < L.nx; x++) L.solid[L.idx(x, y)] = (y == 0 || y > H);
    L.initEquilibrium(1, 0, 0);  // start at rest, let the force spin it up
}

// exact channel profile at (continuous) height y; zero outside the walls
inline double poiseuilleU(const Lattice& L, int H, double y) {
    if (y < 0.5 || y > H + 0.5) return 0.0;
    return L.fx / (2.0 * L.nu) * (y - 0.5) * (H + 0.5 - y);
}

// relative L2 error of column x against the exact profile: sqrt(sum (u - u_exact)^2 / sum u_exact^2)
inline double poiseuilleError(const Lattice& L, int H, int x) {
    double e = 0, n = 0;
    for (int y = 1; y <= H; y++) {
        double ue = poiseuilleU(L, H, y), u = L.ux[L.idx(x, y)];
        e += (u - ue) * (u - ue);
        n += ue * ue;
    }
    return std::sqrt(e / n);
}
