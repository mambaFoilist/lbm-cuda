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
    std::vector<real> uInY;   // inlet profile u(y) instead, if not empty (e.g. parabolic)
    // Inlet/outlet type (when !periodicX):
    //   false: inlet = equilibrium at (1, uIn), outlet = zero-gradient copy. Simple, fine without
    //          side walls, but neither end pins anything: in a walled channel nothing supplies the
    //          pressure drop the walls need, and the flow slowly dies.
    //   true:  non-equilibrium extrapolation (Guo, Zheng & Shi 2002): velocity inlet, pressure outlet
    //          (rho = 1). Each boundary node = equilibrium at the imposed state + the neighbour's
    //          f - f^eq. Second order.
    bool velocityInletPressureOutlet = false;

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

        if (!periodicX && velocityInletPressureOutlet) {
            for (int y = 0; y < ny; y++) {
                if (solid[idx(0, y)] || solid[idx(nx - 1, y)]) continue;
                real ui = uInY.empty() ? uIn : uInY[y];
                extrapolateNonEq(0, 1, y, nullptr, ui, 0);       // inlet: u imposed, rho from inside
                extrapolateNonEq(nx - 1, nx - 2, y, &one, 0, 0);  // outlet: rho = 1, u from inside
            }
        } else if (!periodicX)
        // boundaries: inlet = equilibrium at (rho=1, u=uIn or uInY[y]); outlet = zero-gradient copy
            for (int y = 0; y < ny; y++) {
                real ui = uInY.empty() ? uIn : uInY[y];
                for (int i = 0; i < 9; i++) {
                    f2[fidx(i, 0, y)] = feq(i, 1, ui, 0);
                    f2[fidx(i, nx - 1, y)] = f2[fidx(i, nx - 2, y)];
                }
            }

        f.swap(f2);
    }

    // boundary node (xb, y) from its neighbour (xn, y), on the just-streamed f2:
    //   f_i(b) = f_i^eq(rho_b, u_b) + [f_i(n) - f_i^eq(rho_n, u_n)]
    // rhoB given -> pressure boundary (u_b = u_n); rhoB null -> velocity boundary u_b = (ub, vb), rho_b = rho_n
    static constexpr real one = 1;
    void extrapolateNonEq(int xb, int xn, int y, const real* rhoB, real ub, real vb) {
        real r = 0, mx = 0, my = 0;
        for (int i = 0; i < 9; i++) {
            real fi = f2[fidx(i, xn, y)];
            r += fi; mx += fi * cx[i]; my += fi * cy[i];
        }
        real un = mx / r, vn = my / r;
        real rb = rhoB ? *rhoB : r;
        if (rhoB) { ub = un; vb = vn; }
        for (int i = 0; i < 9; i++)
            f2[fidx(i, xb, y)] = feq(i, rb, ub, vb) + f2[fidx(i, xn, y)] - feq(i, r, un, vn);
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

// Schafer & Turek (1996) benchmark 2D-1: channel 2.2 x 0.41, cylinder D = 0.1 centred at (0.2, 0.2),
// i.e. slightly below the channel middle (0.205), which is why C_L != 0. Parabolic inflow with mean
// speed U (max 1.5 U), no-slip walls, Re = U D / nu = 20, steady.
// Reference (high-accuracy FEM): C_D = 5.57953523384, C_L = 0.010618948146.
//
// In lattice units with N cells per diameter (N a multiple of 10, so the height 4.1 N is whole):
// fluid rows y = 1..H with H = 4.1 N, solid rows y = 0 and y = H + 1 (halfway bounce-back puts the
// walls at y = 0.5 and H + 0.5), nx = 22 N, inlet at x = 0. Cylinder centre (2N + ox, 2N + 0.5 + oy),
// radius N/2: ox, oy shift it by a fraction of a cell, for the convergence study.
// Coefficients: C_D = 2 F_x / (rho U^2 D), C_L = 2 F_y / (rho U^2 D), rho = 1.
struct SchaferTurek {
    static constexpr real CD_REF = 5.57953523384, CL_REF = 0.010618948146;
    int N, H;
    real U, xc, yc, R;
    Lattice L;

    SchaferTurek(int N, real U, real Re = 20, real ox = 0, real oy = 0)
        : N(N), H(41 * N / 10), U(U), xc(2 * N + ox), yc(2 * N + real(0.5) + oy), R(real(0.5) * N),
          L(22 * N, 41 * N / 10 + 2) {
        L.setViscosity(U * N / Re);
        L.periodicX = false;
        L.velocityInletPressureOutlet = true;
        L.fx = L.fy = 0;
        L.uInY.assign(L.ny, 0);
        for (int y = 1; y <= H; y++) {
            real s = y - real(0.5);                           // distance from the bottom wall
            L.uInY[y] = 4 * real(1.5) * U * s * (H - s) / (real(H) * H);
        }
        // start with the inflow profile everywhere: the channel is then already developed and
        // only the flow around the cylinder has to settle
        for (int y = 0; y < L.ny; y++)
            for (int x = 0; x < L.nx; x++) {
                int n = L.idx(x, y);
                L.solid[n] = (y == 0 || y == H + 1 || inCylinder(x, y));
                real u = L.solid[n] ? 0 : L.uInY[y];
                L.rho[n] = 1; L.ux[n] = u; L.uy[n] = 0;
                for (int i = 0; i < 9; i++) L.f[L.fidx(i, x, y)] = feq(i, 1, u, 0);
            }
    }

    bool inCylinder(int x, int y) const { return (x - xc) * (x - xc) + (y - yc) * (y - yc) <= R * R; }

    // force on the cylinder only (not the channel walls); call right after L.step()
    Force force() const { return L.wallForce([this](int x, int y) { return inCylinder(x, y); }); }
    real cd(Force F) const { return 2 * F.x / (U * U * N); }
    real cl(Force F) const { return 2 * F.y / (U * U * N); }
};
