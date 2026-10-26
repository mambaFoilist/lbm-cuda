// lbm.cpp — D2Q9 lattice Boltzmann (BGK), flow past a cylinder, live Win32 window.
//
// Build:  g++ -O3 -march=native -fopenmp src/lbm_viewer.cpp -o lbm_viewer.exe -lgdi32
// Keys:   Space = pause   V = vorticity/speed view   R = reset
// Mouse:  left-drag = draw wall   right-drag = erase wall
//
// Units are lattice units: dx = dt = 1, c_s^2 = 1/3.
// Everything below maps to 4 equations:
//   1. moments:      rho = sum f_i,   rho*u = sum f_i * c_i
//   2. equilibrium:  f_i^eq = w_i rho (1 + 3 c.u + 4.5 (c.u)^2 - 1.5 u.u)
//   3. collide:      f_i <- f_i - (f_i - f_i^eq) / tau
//   4. stream:       f_i(x + c_i, t+1) = f_i(x, t)
// Chapman-Enskog shows this recovers Navier-Stokes with nu = c_s^2 (tau - 1/2), p = c_s^2 rho.

#include <windows.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

// ---------------- simulation parameters ----------------
constexpr int NX = 400, NY = 100;     // lattice size
constexpr int SCALE = 3;              // screen pixels per lattice cell
constexpr int STEPS_PER_FRAME = 20;
constexpr float U0 = 0.1f;            // inlet speed (Ma = U0/c_s ~ 0.17, keep < ~0.3)
constexpr float RE = 100.0f;          // Reynolds number based on cylinder diameter
constexpr int CYL_X = NX / 5, CYL_Y = NY / 2 + 2, CYL_R = NY / 9;  // +2 offset breaks symmetry -> shedding

// Not const so other cases (e.g. the Poiseuille check) can pick their own viscosity via setViscosity().
float NU = U0 * 2 * CYL_R / RE;   // Re = U D / nu  ->  nu = U D / Re
float TAU = 3.0f * NU + 0.5f;     // nu = (tau - 1/2)/3  ->  tau. Must stay > 0.5
float OMEGA = 1.0f / TAU;

void setViscosity(float nu) { NU = nu; TAU = 3.0f * nu + 0.5f; OMEGA = 1.0f / TAU; }

// Optional extras, both off for the cylinder:
//   body force density (FX, FY), added with Guo's scheme in the collision
//   PERIODIC_X: left/right edges wrap around instead of inlet/outlet
float FX = 0.0f, FY = 0.0f;
bool PERIODIC_X = false;

// ---------------- D2Q9 lattice ----------------
//   6 2 5
//   3 0 1
//   7 4 8
const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const float w[9] = {4.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 9, 1.f / 36, 1.f / 36, 1.f / 36, 1.f / 36};
const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};  // direction pointing the other way

// ---------------- fields ----------------
std::vector<float> f(9 * NX * NY), f2(9 * NX * NY);  // populations (two buffers for streaming)
std::vector<float> rho(NX * NY), ux(NX * NY), uy(NX * NY);
std::vector<uint8_t> solid(NX * NY);

inline int idx(int x, int y) { return y * NX + x; }
inline int fidx(int i, int x, int y) { return (i * NY + y) * NX + x; }

inline float feq(int i, float r, float u, float v) {
    float cu = 3.0f * (cx[i] * u + cy[i] * v);
    return w[i] * r * (1.0f + cu + 0.5f * cu * cu - 1.5f * (u * u + v * v));
}

bool inShape(int x, int y) {
    float dx = x - CYL_X, dy = y - CYL_Y;

    // Circle
    //return dx*dx + dy*dy <= CYL_R*CYL_R;

    // Square:      return std::abs(dx) <= CYL_R && std::abs(dy) <= CYL_R;
    // Ellipse:     return (dx*dx)/(2.0f*CYL_R*CYL_R) + (dy*dy)/(0.5f*CYL_R*CYL_R) <= 1.0f;
    //Flat plate:  
    return std::abs(dx) <= 1 && std::abs(dy) <= CYL_R;
}

void reset() {
    for (int y = 0; y < NY; y++)
        for (int x = 0; x < NX; x++) {
            //int dx = x - CYL_X, dy = y - CYL_Y;
            solid[idx(x, y)] = inShape(x,y);
            rho[idx(x, y)] = 1.0f;
            ux[idx(x, y)] = solid[idx(x, y)] ? 0.0f : U0;
            uy[idx(x, y)] = 0.0f;
            for (int i = 0; i < 9; i++) f[fidx(i, x, y)] = feq(i, 1.0f, ux[idx(x, y)], 0.0f);
        }
}

// Channel for the Poiseuille check: fluid rows y = 1..H, solid at y = 0 and y > H, periodic in x,
// driven by a uniform body force. With halfway bounce-back the walls sit at y = 0.5 and y = H + 0.5,
// and the exact steady profile is u(y) = FX / (2 nu) * (y - 0.5) * (H + 0.5 - y)  (rho = 1).
// FX is chosen so the centerline speed is umax.
void setupChannel(int H, float umax) {
    PERIODIC_X = true;
    FX = 8.0f * NU * umax / (float(H) * H);  // u_max = F H^2 / (8 nu)
    FY = 0.0f;
    for (int y = 0; y < NY; y++)
        for (int x = 0; x < NX; x++) {
            solid[idx(x, y)] = (y == 0 || y > H);
            rho[idx(x, y)] = 1.0f;
            ux[idx(x, y)] = uy[idx(x, y)] = 0.0f;  // start at rest, let the force spin it up
            for (int i = 0; i < 9; i++) f[fidx(i, x, y)] = feq(i, 1.0f, 0.0f, 0.0f);
        }
}

void step() {
    const bool forced = FX != 0.0f || FY != 0.0f;  // unforced path stays exactly the plain BGK update

    // 1-3: compute moments, then BGK collision (in place)
    //   With a force (Guo et al. 2002): u = (sum f_i c_i + F/2) / rho, and after relaxing add
    //   S_i = (1 - omega/2) w_i [3 (c_i - u).F + 9 (c_i.u)(c_i.F)]. Keeps the scheme second order.
#pragma omp parallel for
    for (int y = 0; y < NY; y++)
        for (int x = 0; x < NX; x++) {
            if (solid[idx(x, y)]) continue;
            float r = 0, mx = 0, my = 0;
            for (int i = 0; i < 9; i++) {
                float fi = f[fidx(i, x, y)];
                r += fi;
                mx += fi * cx[i];
                my += fi * cy[i];
            }
            float u = mx / r, v = my / r;
            if (forced) { u += 0.5f * FX / r; v += 0.5f * FY / r; }
            rho[idx(x, y)] = r; ux[idx(x, y)] = u; uy[idx(x, y)] = v;
            for (int i = 0; i < 9; i++) {
                float& fi = f[fidx(i, x, y)];
                fi += OMEGA * (feq(i, r, u, v) - fi);
                if (forced) {
                    float cF = cx[i] * FX + cy[i] * FY, cu = cx[i] * u + cy[i] * v;
                    fi += (1.0f - 0.5f * OMEGA) * w[i] * (3.0f * (cF - u * FX - v * FY) + 9.0f * cu * cF);
                }
            }
        }

    // 4: streaming ("pull": each node grabs what arrives from x - c_i).
    //    If the source is a wall, the population that left toward the wall
    //    bounces back: f_i(x) = f_opp(i)(x)  -> no-slip wall (halfway bounce-back).
    //    Top/bottom are periodic; left/right too if PERIODIC_X.
#pragma omp parallel for
    for (int y = 0; y < NY; y++)
        for (int x = 0; x < NX; x++) {
            if (solid[idx(x, y)]) continue;
            for (int i = 0; i < 9; i++) {
                int xs = x - cx[i];
                if (PERIODIC_X) xs = (xs + NX) % NX;
                int ys = (y - cy[i] + NY) % NY;
                if (xs < 0 || xs >= NX)        f2[fidx(i, x, y)] = f[fidx(i, x, y)];  // fixed by BCs below
                else if (solid[idx(xs, ys)])   f2[fidx(i, x, y)] = f[fidx(opp[i], x, y)];
                else                           f2[fidx(i, x, y)] = f[fidx(i, xs, ys)];
            }
        }

    // boundaries: inlet = equilibrium at (rho=1, u=U0); outlet = zero-gradient copy
    if (!PERIODIC_X)
    for (int y = 0; y < NY; y++)
        for (int i = 0; i < 9; i++) {
            f2[fidx(i, 0, y)] = feq(i, 1.0f, U0, 0.0f);
            f2[fidx(i, NX - 1, y)] = f2[fidx(i, NX - 2, y)];
        }

    f.swap(f2);
}

// ---------------- rendering ----------------
uint32_t pixels[NX * NY];
bool showVorticity = true, paused = false;

uint32_t rgb(float r, float g, float b) {
    auto c = [](float v) { return (uint32_t)(v < 0 ? 0 : v > 1 ? 255 : v * 255); };
    return (c(r) << 16) | (c(g) << 8) | c(b);
}

void render() {
#pragma omp parallel for
    for (int y = 0; y < NY; y++)
        for (int x = 0; x < NX; x++) {
            uint32_t& p = pixels[idx(x, y)];
            if (solid[idx(x, y)]) { p = rgb(0.25f, 0.25f, 0.25f); continue; }
            if (showVorticity) {
                // curl u = d(uy)/dx - d(ux)/dy, central differences
                int xl = x > 0 ? x - 1 : x, xr = x < NX - 1 ? x + 1 : x;
                int yd = (y - 1 + NY) % NY, yu = (y + 1) % NY;
                float curl = (uy[idx(xr, y)] - uy[idx(xl, y)]) - (ux[idx(x, yu)] - ux[idx(x, yd)]);
                float s = curl / 0.04f;  // diverging map: blue <- 0 -> red
                p = s > 0 ? rgb(1, 1 - s, 1 - s) : rgb(1 + s, 1 + s, 1);
            } else {
                float s = std::sqrt(ux[idx(x, y)] * ux[idx(x, y)] + uy[idx(x, y)] * uy[idx(x, y)]) / (1.6f * U0);
                p = rgb(1.5f * s - 0.5f, std::sin(3.14159f * s), 1.0f - 1.5f * s);
            }
        }
}

// ---------------- window / input ----------------
void paint(HWND hwnd) {
    BITMAPINFO bmi = {};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = NX;
    bmi.bmiHeader.biHeight = NY;  // positive = bottom-up, so lattice y=0 is at the bottom
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    HDC dc = GetDC(hwnd);
    StretchDIBits(dc, 0, 0, NX * SCALE, NY * SCALE, 0, 0, NX, NY, pixels, &bmi, DIB_RGB_COLORS, SRCCOPY);
    ReleaseDC(hwnd, dc);
}

void brush(LPARAM lp, bool makeSolid) {
    int mx = (short)LOWORD(lp) / SCALE, my = NY - 1 - (short)HIWORD(lp) / SCALE;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++) {
            int x = mx + dx, y = my + dy;
            if (x < 1 || x >= NX - 1 || y < 0 || y >= NY) continue;
            solid[idx(x, y)] = makeSolid;
            if (!makeSolid) {  // freshly opened cell starts as fluid at rest
                ux[idx(x, y)] = uy[idx(x, y)] = 0;
                for (int i = 0; i < 9; i++) f[fidx(i, x, y)] = feq(i, 1.0f, 0, 0);
            }
        }
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_DESTROY: PostQuitMessage(0); return 0;
    case WM_KEYDOWN:
        if (wp == VK_SPACE) paused = !paused;
        if (wp == 'V') showVorticity = !showVorticity;
        if (wp == 'R') reset();
        return 0;
    case WM_LBUTTONDOWN: case WM_RBUTTONDOWN: case WM_MOUSEMOVE:
        if (wp & MK_LBUTTON) brush(lp, true);
        else if (wp & MK_RBUTTON) brush(lp, false);
        return 0;
    }
    return DefWindowProc(hwnd, msg, wp, lp);
}

int main() {
    std::printf("D2Q9 LBM  %dx%d  Re=%.0f  U0=%.3f  nu=%.4f  tau=%.4f\n", NX, NY, RE, U0, NU, TAU);
    if (TAU < 0.51f) std::printf("WARNING: tau too close to 0.5, will likely blow up. Lower Re or raise resolution.\n");
    reset();

    WNDCLASSA wc = {};
    wc.lpfnWndProc = WndProc;
    wc.hInstance = GetModuleHandle(nullptr);
    wc.hCursor = LoadCursor(nullptr, IDC_CROSS);
    wc.lpszClassName = "LBM";
    RegisterClassA(&wc);
    RECT rc = {0, 0, NX * SCALE, NY * SCALE};
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME, FALSE);
    HWND hwnd = CreateWindowA("LBM", "LBM", (WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME) | WS_VISIBLE,
                              CW_USEDEFAULT, CW_USEDEFAULT, rc.right - rc.left, rc.bottom - rc.top,
                              nullptr, nullptr, wc.hInstance, nullptr);

    long long t = 0, stepsSinceTitle = 0;
    DWORD lastTitle = GetTickCount();
    for (;;) {
        MSG m;
        while (PeekMessage(&m, nullptr, 0, 0, PM_REMOVE)) {
            if (m.message == WM_QUIT) return 0;
            TranslateMessage(&m);
            DispatchMessage(&m);
        }
        if (!paused) {
            for (int s = 0; s < STEPS_PER_FRAME; s++) step();
            t += STEPS_PER_FRAME;
            stepsSinceTitle += STEPS_PER_FRAME;
            if (std::isnan(rho[idx(NX / 2, NY / 2)])) {
                std::printf("Diverged at step %lld (NaN). Resetting.\n", t);
                reset();
                t = 0;
            }
        } else Sleep(16);
        render();
        paint(hwnd);

        DWORD now = GetTickCount();
        if (now - lastTitle >= 500) {
            double mlups = stepsSinceTitle * double(NX * NY) / ((now - lastTitle) * 1e3);
            char title[160];
            std::snprintf(title, sizeof title, "LBM  step %lld  |  %.1f MLUPS  |  %s%s", t, mlups,
                          showVorticity ? "vorticity" : "speed", paused ? "  [PAUSED]" : "");
            SetWindowTextA(hwnd, title);
            lastTitle = now;
            stepsSinceTitle = 0;
        }
    }
}
