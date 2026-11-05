// sweep_poiseuille_tau.cpp — research question 1, part 1: what does tau really do in a channel?
//
// Build + run (from C:\Users\diego\lbm):
//   g++ -O3 -march=native -fopenmp tests/sweep_poiseuille_tau.cpp -o sweep_poiseuille_tau.exe
//   ./sweep_poiseuille_tau.exe            (a few seconds: every lattice is a single 1 x (H+2) column)
//
// Same body-force channel as check_poiseuille.cpp (setupChannel in lbm_core.h), at
//   tau in {0.55, 0.6, 0.7, 0.8, 0.933, 1.0, 1.2, 1.5, 2.0},  H = 16 and 32,  u_max = 0.1.
//
// Idea: BGK + halfway bounce-back with Guo forcing gives a steady profile that is an EXACT parabola
// (the bulk scheme has no error for a quadratic), but the parabola's zeros are not exactly at the
// nominal walls y = 0.5 and H + 0.5. So fit u = a y^2 + b y + c to the simulated profile and read off
//   * viscosity:       steady momentum  rho nu u'' = -F   ->  nu_meas = -F / (2 a rho)
//                      compare with nu_theory = (tau - 1/2) / 3  (Chapman-Enskog)
//   * wall location:   zeros of the fit, distance H_eff = sqrt(b^2 - 4ac) / |a|
//                      compare with H_eff^2 = H^2 + (16 Lambda - 3)/3, Lambda = (tau - 1/2)^2
//                      (He et al. 1997; Companion ch. 6.3). H_eff = H only at the "magic"
//                      tau = 1/2 + sqrt(3/16) = 0.933: there the wall is exactly halfway.
//   * slip:            u at the nominal wall, relative to the centreline speed
//   * fit residual:    should be round-off if the profile really is a parabola
// The fit is done in eta = y - (H+1)/2 (centred) so the normal equations stay well conditioned.
//
// Also: L2 error vs H at tau = 0.8 and 1.0 (PLAN.md figure 4), and the H = 16 profiles (figure 1).
//
// Output (runs/):
//   poiseuille_tau.csv          one row per (H, tau)
//   poiseuille_profiles.csv     u_sim and u_exact(nominal walls) for every node, H = 16
//   poiseuille_convergence.csv  L2 error vs H at tau = 0.8, 1.0

#include "../src/lbm_core.h"
#include "run_util.h"
#include <cstdio>
#include <string>
#include <vector>

const real UMAX = 0.1;

struct Fit { double a, b, c, resid; };  // u = a eta^2 + b eta + c, resid = rms misfit / u_center

// least-squares parabola through (eta_k, u_k): 3x3 normal equations, Cramer's rule
Fit fitParabola(const std::vector<double>& eta, const std::vector<double>& u) {
    double S[5] = {}, T[3] = {};  // S[p] = sum eta^p, T[p] = sum eta^p u
    for (size_t k = 0; k < eta.size(); k++) {
        double e = 1;
        for (int p = 0; p < 5; p++) {
            S[p] += e;
            if (p < 3) T[p] += e * u[k];
            e *= eta[k];
        }
    }
    // [S4 S3 S2][a]   [T2]
    // [S3 S2 S1][b] = [T1]
    // [S2 S1 S0][c]   [T0]
    auto det3 = [](double m00, double m01, double m02, double m10, double m11, double m12, double m20,
                   double m21, double m22) {
        return m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20) + m02 * (m10 * m21 - m11 * m20);
    };
    double D = det3(S[4], S[3], S[2], S[3], S[2], S[1], S[2], S[1], S[0]);
    Fit f;
    f.a = det3(T[2], S[3], S[2], T[1], S[2], S[1], T[0], S[1], S[0]) / D;
    f.b = det3(S[4], T[2], S[2], S[3], T[1], S[1], S[2], T[0], S[0]) / D;
    f.c = det3(S[4], S[3], T[2], S[3], S[2], T[1], S[2], S[1], T[0]) / D;
    double r2 = 0;
    for (size_t k = 0; k < eta.size(); k++) {
        double d = u[k] - (f.a * eta[k] * eta[k] + f.b * eta[k] + f.c);
        r2 += d * d;
    }
    f.resid = std::sqrt(r2 / eta.size()) / f.c;
    return f;
}

// run the channel to steady state (same criterion as check_poiseuille.cpp); returns steps taken, -1 if not
long long runToSteady(Lattice& L, int H) {
    const int CHECK = 500;
    const long long maxSteps = (long long)(50.0 * H * H / (L.nu * 9.87)) + 10 * CHECK;
    std::vector<double> prev(H + 1, 0.0);
    for (long long t = 1; t <= maxSteps; t++) {
        L.step();
        if (t % CHECK) continue;
        double d = 0, n = 0;
        for (int y = 1; y <= H; y++) {
            double u = L.ux[L.idx(0, y)];
            d += (u - prev[y]) * (u - prev[y]);
            n += u * u;
            prev[y] = u;
        }
        if (std::isnan(n)) return -1;
        if (std::sqrt(d / n) < 1e-12) return t;
    }
    return -1;
}

int main() {
    const std::vector<double> taus = {0.55, 0.6, 0.7, 0.8, 0.5 + std::sqrt(3.0 / 16.0), 1.0, 1.2, 1.5, 2.0};
    const std::string meta = "D2Q9 BGK, halfway bounce-back, Guo forcing, periodic x, u_max=0.1, double";
    Stopwatch sw;
    bool allOk = true;

    FILE* fs = openCsv("runs/poiseuille_tau.csv",
                       "H,tau,nu_theory,nu_meas,nu_rel_err,H_eff_meas,H_eff_theory,wall_shift_meas,"
                       "wall_shift_theory,slip_meas,slip_theory,fit_resid,l2_error,steps",
                       meta);
    FILE* fp = openCsv("runs/poiseuille_profiles.csv", "H,tau,y,u_sim,u_exact", meta);

    std::printf("tau sweep, u_max = %.2f   (wall shift = (H_eff - H)/2 per wall, in cells; slip = u_wall / u_centre)\n", UMAX);
    std::printf("   H    tau    nu_meas/nu_th - 1   H_eff meas   H_eff theory   wall shift   slip meas   slip theory   fit resid\n");
    for (int H : {16, 32}) {
        for (double tau : taus) {
            Lattice L(1, H + 2);
            L.setViscosity((tau - 0.5) / 3);
            setupChannel(L, H, UMAX);
            long long steps = runToSteady(L, H);
            if (steps < 0) { std::printf("  H=%d tau=%.3f did NOT reach steady state\n", H, tau); allOk = false; }

            std::vector<double> eta, u;
            double rhoMean = 0;
            for (int y = 1; y <= H; y++) {
                eta.push_back(y - 0.5 * (H + 1));
                u.push_back(L.ux[L.idx(0, y)]);
                rhoMean += L.rho[L.idx(0, y)] / H;
            }
            Fit f = fitParabola(eta, u);
            double nuTh = (tau - 0.5) / 3;
            double nuMeas = -L.fx / (2 * f.a * rhoMean);
            double Heff = std::sqrt(f.b * f.b - 4 * f.a * f.c) / std::fabs(f.a);
            double Lam = (tau - 0.5) * (tau - 0.5);
            double HeffTh = std::sqrt(double(H) * H + (16 * Lam - 3) / 3);
            // velocity of the fitted parabola at the nominal wall eta = H/2, relative to the centre
            double slip = (f.a * H * H / 4 + f.c) / f.c;            // (b ~ 0 by symmetry)
            double slipTh = (HeffTh * HeffTh - double(H) * H) / (HeffTh * HeffTh);
            double l2 = poiseuilleError(L, H, 0);

            std::printf("  %2d  %5.3f   %+.2e          %9.5f   %9.5f      %+8.5f   %+.2e   %+.2e     %.1e\n", H,
                        tau, nuMeas / nuTh - 1, Heff, HeffTh, (Heff - H) / 2, slip, slipTh, f.resid);
            std::fprintf(fs, "%d,%.6f,%.12e,%.12e,%.3e,%.10f,%.10f,%.10f,%.10f,%.6e,%.6e,%.3e,%.6e,%lld\n", H, tau,
                         nuTh, nuMeas, nuMeas / nuTh - 1, Heff, HeffTh, (Heff - H) / 2, (HeffTh - H) / 2, slip,
                         slipTh, f.resid, l2, steps);
            if (H == 16)
                for (int y = 0; y <= H + 1; y++)
                    std::fprintf(fp, "%d,%.6f,%d,%.12e,%.12e\n", H, tau, y, L.ux[L.idx(0, y)], poiseuilleU(L, H, y));

            // the wall-location formula should hold to round-off; the viscosity too
            allOk &= std::fabs(nuMeas / nuTh - 1) < 1e-8 && std::fabs(Heff - HeffTh) < 1e-6;
        }
        std::printf("\n");
    }
    std::fclose(fs);
    std::fclose(fp);

    // ---- L2 error vs H (second order) at two tau ----
    FILE* fc = openCsv("runs/poiseuille_convergence.csv", "tau,H,l2_error,steps", meta);
    std::printf("L2 error vs H\n   tau     H    L2 error    order\n");
    for (double tau : {0.8, 1.0}) {
        double ePrev = 0;
        int hPrev = 0;
        for (int H : {4, 8, 16, 32, 64}) {
            Lattice L(1, H + 2);
            L.setViscosity((tau - 0.5) / 3);
            setupChannel(L, H, UMAX);
            long long steps = runToSteady(L, H);
            double e = poiseuilleError(L, H, 0);
            std::fprintf(fc, "%.6f,%d,%.6e,%lld\n", tau, H, e, steps);
            std::printf("  %.2f  %4d   %.3e", tau, H, e);
            if (hPrev) std::printf("   %.3f", std::log(ePrev / e) / std::log(double(H) / hPrev));
            std::printf("\n");
            ePrev = e; hPrev = H;
        }
    }
    std::fclose(fc);

    std::printf("\n(%.1f s)  wrote runs/poiseuille_tau.csv, runs/poiseuille_profiles.csv, runs/poiseuille_convergence.csv\n",
                sw.seconds());
    std::printf("[%s] nu_meas = (tau - 1/2)/3 and H_eff matches H^2 + (16 Lambda - 3)/3 at every tau\n",
                allOk ? "PASS" : "FAIL");
    return allOk ? 0 : 1;
}
