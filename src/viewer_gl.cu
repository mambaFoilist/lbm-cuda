// viewer_gl.cu — live OpenGL viewer for the GPU solver (lbm_gpu.cuh), single precision.
//
// Build:  .\build_viewer_gl.bat        Run:  .\viewer_gl.exe
// Keys:   Space = pause   N = one step while paused   R = reset   C = switch case
//         V = vorticity / speed   Up/Down = Reynolds number x1.25 / /1.25
//         + / - = steps per frame x2 / /2   [ / ] = colour contrast   Esc = quit
// Mouse:  left-drag = draw wall   right-drag = erase wall
//
// Cases: "cylinder" (1024 x 256, periodic top/bottom, equilibrium inlet; same setup as the CPU
// viewer, 2.6x the resolution) and "Schafer-Turek channel" (N = 40, walls, parabolic velocity
// inlet, pressure outlet: the benchmark geometry, at Re = 100 by default so it sheds).
//
// Each frame: k solver steps, then a CUDA kernel turns rho/u into one float per cell (vorticity
// or speed), which is copied to the CPU and uploaded as an OpenGL texture, coloured by a fragment
// shader. That copy is ~1 MB per frame: nothing next to the simulation itself.
//
//   .\viewer_gl.exe --interop   instead writes the field straight into an OpenGL pixel buffer
//                               (CUDA-OpenGL interop), so it never leaves the GPU. Off by default:
//   the first test run of the interop version (OpenGL forced onto the NVIDIA GPU of this Optimus
//   laptop + a buffer shared between the CUDA and OpenGL drivers) blue-screened Windows
//   (bugcheck 0x139, 2026-10-27). A driver bug, not ours, but not worth risking. Interop only works
//   when OpenGL runs on the NVIDIA GPU: set viewer_gl.exe to "High performance" in Windows
//   Settings > System > Display > Graphics first; otherwise it falls back to the copy.

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <glad/gl.h>  // before anything else that might pull in GL/gl.h
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

#include "lbm_gpu.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

// No NvOptimusEnablement export here (it forces OpenGL onto the NVIDIA GPU): that was part of the
// setup that crashed Windows. OpenGL runs wherever Windows puts it; CUDA always uses the NVIDIA GPU.

using G = LatticeGPU<float>;

// ---------------- kernels ----------------
constexpr float SOLID_MARK = 1e30f;  // field value the shader draws as wall

// mode 0: vorticity  w = dv/dx - du/dy (central differences), in units of U/D
// mode 1: speed |u|, in units of U
__global__ void fieldKernel(gpu::Params<float> p, int mode, float norm, float* out) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= p.nx * p.ny) return;
    if (p.solid[n]) { out[n] = SOLID_MARK; return; }
    int x = n % p.nx, y = n / p.nx;
    if (mode == 1) { out[n] = norm * sqrtf(p.ux[n] * p.ux[n] + p.uy[n] * p.uy[n]); return; }
    int xl = x - 1, xr = x + 1;
    if (p.periodicX) { xl = (xl + p.nx) % p.nx; xr = xr % p.nx; }
    else { xl = max(xl, 0); xr = min(xr, p.nx - 1); }
    int yd = (y - 1 + p.ny) % p.ny, yu = (y + 1) % p.ny;
    auto U = [&](int m) { return p.solid[m] ? 0.0f : p.ux[m]; };  // walls don't move
    auto V = [&](int m) { return p.solid[m] ? 0.0f : p.uy[m]; };
    float dvdx = 0.5f * (V(y * p.nx + xr) - V(y * p.nx + xl));
    float dudy = 0.5f * (U(yu * p.nx + x) - U(yd * p.nx + x));
    out[n] = norm * (dvdx - dudy);
}

// mouse brush: cells within r of (cx, cy) become wall, or (erase) fluid at rest
__global__ void paintKernel(gpu::Params<float> p, uint8_t* solid, float* f, float cx, float cy, float r, bool makeSolid) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= p.nx * p.ny) return;
    int x = n % p.nx, y = n / p.nx;
    if ((x - cx) * (x - cx) + (y - cy) * (y - cy) > r * r) return;
    if (!p.periodicX && (x < 2 || x > p.nx - 3)) return;  // keep inlet/outlet columns fluid
    if (makeSolid) { solid[n] = 1; return; }
    if (!solid[n]) return;
    solid[n] = 0;
    for (int i = 0; i < 9; i++) f[gpu::fidx(i, x, y, p.nx, p.ny)] = gpu::feq<float>(i, 1, 0, 0);
    p.ux[n] = p.uy[n] = 0;
}

// ---------------- shaders ----------------
const char* VERT = R"(#version 330 core
out vec2 uv;
void main() {  // one triangle that covers the screen
    vec2 pos = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
})";

const char* FRAG = R"(#version 330 core
in vec2 uv;
out vec4 color;
uniform sampler2D field;
uniform int mode;      // 0 vorticity, 1 speed
uniform float scale;   // field value at full colour
vec3 coolwarm(float s) {  // s in [-1, 1]: blue - white - red
    vec3 cold = vec3(0.23, 0.30, 0.75), hot = vec3(0.71, 0.02, 0.15), mid = vec3(0.97);
    return s < 0.0 ? mix(mid, cold, -s) : mix(mid, hot, s);
}
vec3 turbo(float t) {  // Google's Turbo colormap, polynomial fit
    const vec4 kR = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kG = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kB = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kR2 = vec2(-152.94239396, 59.28637943);
    const vec2 kG2 = vec2(4.27729857, 2.82956604);
    const vec2 kB2 = vec2(-89.90310912, 27.34824973);
    vec4 v4 = vec4(1.0, t, t * t, t * t * t);
    vec2 v2 = v4.zw * v4.z;
    return vec3(dot(v4, kR) + dot(v2, kR2), dot(v4, kG) + dot(v2, kG2), dot(v4, kB) + dot(v2, kB2));
}
void main() {
    float v = texture(field, uv).r;
    if (v > 1e29) { color = vec4(0.30, 0.30, 0.32, 1.0); return; }
    if (mode == 0) color = vec4(coolwarm(clamp(v / scale, -1.0, 1.0)), 1.0);
    else           color = vec4(turbo(clamp(v / scale, 0.0, 1.0)), 1.0);
})";

GLuint compile(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        std::fprintf(stderr, "shader error:\n%s\n", log);
        std::exit(1);
    }
    return s;
}

// ---------------- state ----------------
struct Viewer {
    // case
    int caseId = 0;              // 0 cylinder, 1 Schafer-Turek channel
    std::unique_ptr<G> g;
    int nx = 0, ny = 0;
    double U = 0, D = 0, Re = 0; // inflow speed, body size (for Re and the coefficients)
    long long step = 0;

    // view
    int mode = 0, stepsPerFrame = 32;
    float scale[2] = {4.0f, 1.6f};
    bool paused = false, stepOnce = false;
    bool tryInterop = false;     // --interop on the command line

    // GL
    GLuint prog = 0, vao = 0, tex = 0, pbo = 0;
    cudaGraphicsResource* res = nullptr;  // pbo as seen by CUDA; null -> no interop
    float* dField = nullptr;              // fallback path: field on GPU, copied via host
    std::vector<float> hField;
} V;

void buildCase(double Re) {
    if (V.caseId == 0) {
        const int NX = 1024, NY = 256, R = NY / 9, CX = NX / 5, CY = NY / 2;
        Lattice L(NX, NY);
        V.U = 0.1; V.D = 2 * R; V.Re = Re;
        L.setViscosity(V.U * V.D / Re);
        setupFlowPast(L, V.U, [&](int x, int y) { return (x - CX) * (x - CX) + (y - CY) * (y - CY) <= R * R; });
        kickWake(L, CX + R, CY, R, 0.1 * V.U);
        V.g = std::make_unique<G>(L);
    } else {
        SchaferTurek st(40, 0.05, Re);
        V.U = st.U; V.D = st.N; V.Re = Re;
        kickWake(st.L, int(st.xc + st.R), int(st.yc), int(st.R), 0.1 * st.U);  // shed right away
        V.g = std::make_unique<G>(st.L);
        V.g->setForceBody([&](int x, int y) { return st.inCylinder(x, y); });  // cylinder, not the walls
    }
    V.nx = V.g->nx; V.ny = V.g->ny;
    V.step = 0;
    V.g->step();  // first step only collides; afterwards the moments exist
    V.step = 1;
    std::printf("%s  %dx%d  Re=%.0f  tau=%.4f\n", V.caseId ? "Schafer-Turek channel" : "cylinder", V.nx, V.ny,
                Re, 3 * V.U * V.D / Re + 0.5);
}

// texture + pixel buffer for the current grid size (re-made when the case changes)
void buildTexture() {
    if (V.res) { cudaGraphicsUnregisterResource(V.res); V.res = nullptr; }
    if (V.dField) { cudaFree(V.dField); V.dField = nullptr; }
    if (V.tex) glDeleteTextures(1, &V.tex);
    if (V.pbo) glDeleteBuffers(1, &V.pbo);

    glGenTextures(1, &V.tex);
    glBindTexture(GL_TEXTURE_2D, V.tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, V.nx, V.ny, 0, GL_RED, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);  // nearest: walls stay crisp
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glGenBuffers(1, &V.pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, V.pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_t(V.nx) * V.ny * sizeof(float), nullptr, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    if (V.tryInterop &&
        cudaGraphicsGLRegisterBuffer(&V.res, V.pbo, cudaGraphicsRegisterFlagsWriteDiscard) != cudaSuccess) {
        cudaGetLastError();  // clear it; use the copy-through-CPU path instead
        V.res = nullptr;
        std::printf("CUDA-OpenGL interop not available (is OpenGL on the NVIDIA GPU?): using the copy\n");
    }
    if (!V.res) {  // default path: field on the GPU, copied through the CPU into the texture
        CUDA_CHECK(cudaMalloc(&V.dField, size_t(V.nx) * V.ny * sizeof(float)));
        V.hField.resize(size_t(V.nx) * V.ny);
    }
}

void reset() { buildCase(V.Re); buildTexture(); }

void uploadField() {
    int N = V.nx * V.ny, blocks = (N + 255) / 256;
    float norm = V.mode == 0 ? float(V.D / V.U) : float(1 / V.U);
    if (V.res) {
        float* ptr;
        size_t bytes;
        CUDA_CHECK(cudaGraphicsMapResources(1, &V.res));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&ptr, &bytes, V.res));
        fieldKernel<<<blocks, 256>>>(V.g->p, V.mode, norm, ptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaGraphicsUnmapResources(1, &V.res));
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, V.pbo);
        glBindTexture(GL_TEXTURE_2D, V.tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, V.nx, V.ny, GL_RED, GL_FLOAT, nullptr);  // from the pbo
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    } else {
        fieldKernel<<<blocks, 256>>>(V.g->p, V.mode, norm, V.dField);
        CUDA_CHECK(cudaMemcpy(V.hField.data(), V.dField, N * sizeof(float), cudaMemcpyDeviceToHost));
        glBindTexture(GL_TEXTURE_2D, V.tex);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, V.nx, V.ny, GL_RED, GL_FLOAT, V.hField.data());
    }
}

// largest nx:ny rectangle centred in a w x h area
void letterbox(int w, int h, int& x0, int& y0, int& vw, int& vh) {
    double a = double(V.nx) / V.ny;
    if (w > a * h) { vh = h; vw = int(a * h); } else { vw = w; vh = int(w / a); }
    x0 = (w - vw) / 2; y0 = (h - vh) / 2;
}

void setRe(double Re) {
    double nu = V.U * V.D / Re;
    if (3 * nu + 0.5 < 0.505) { std::printf("Re %.0f would put tau below 0.505 (unstable): not changed\n", Re); return; }
    V.Re = Re;
    V.g->setViscosity(nu);
    std::printf("Re = %.0f  (tau = %.4f)\n", Re, 3 * nu + 0.5);
}

void onKey(GLFWwindow* w, int key, int, int action, int) {
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    switch (key) {
    case GLFW_KEY_ESCAPE: glfwSetWindowShouldClose(w, 1); break;
    case GLFW_KEY_SPACE: V.paused = !V.paused; break;
    case GLFW_KEY_N: if (V.paused) V.stepOnce = true; break;
    case GLFW_KEY_R: reset(); break;
    case GLFW_KEY_C: V.caseId ^= 1; buildCase(V.caseId ? 100 : 150); buildTexture(); break;
    case GLFW_KEY_V: V.mode ^= 1; break;
    case GLFW_KEY_UP: setRe(V.Re * 1.25); break;
    case GLFW_KEY_DOWN: setRe(V.Re / 1.25); break;
    case GLFW_KEY_EQUAL: case GLFW_KEY_KP_ADD: V.stepsPerFrame = std::min(V.stepsPerFrame * 2, 4096); break;
    case GLFW_KEY_MINUS: case GLFW_KEY_KP_SUBTRACT: V.stepsPerFrame = std::max(V.stepsPerFrame / 2, 1); break;
    case GLFW_KEY_LEFT_BRACKET: V.scale[V.mode] *= 1.25f; break;   // wider range = less contrast
    case GLFW_KEY_RIGHT_BRACKET: V.scale[V.mode] /= 1.25f; break;
    }
}

int main(int argc, char** argv) {
    for (int a = 1; a < argc; a++)
        if (std::string(argv[a]) == "--interop") V.tryInterop = true;
    if (!glfwInit()) { std::fprintf(stderr, "glfwInit failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win = glfwCreateWindow(1600, 460, "LBM GPU", nullptr, nullptr);
    if (!win) { std::fprintf(stderr, "could not open an OpenGL 3.3 window\n"); return 1; }
    glfwMakeContextCurrent(win);
    if (!gladLoadGL(glfwGetProcAddress)) { std::fprintf(stderr, "gladLoadGL failed\n"); return 1; }
    glfwSwapInterval(1);  // vsync
    glfwSetKeyCallback(win, onKey);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("OpenGL: %s | CUDA: %s\n", (const char*)glGetString(GL_RENDERER), prop.name);
    std::printf("Space pause, N step, R reset, C case, V field, Up/Down Re, +/- speed, [ ] contrast,\n"
                "left/right drag = draw/erase walls, Esc quit\n\n");

    V.prog = glCreateProgram();
    glAttachShader(V.prog, compile(GL_VERTEX_SHADER, VERT));
    glAttachShader(V.prog, compile(GL_FRAGMENT_SHADER, FRAG));
    glLinkProgram(V.prog);
    glGenVertexArrays(1, &V.vao);  // core profile needs one bound, even with no vertex data
    GLint locMode = glGetUniformLocation(V.prog, "mode"), locScale = glGetUniformLocation(V.prog, "scale");

    V.Re = 150;
    reset();

    double tLast = glfwGetTime();
    long long stepsSince = 0;
    Force F = {0, 0};
    int frame = 0;
    while (!glfwWindowShouldClose(win)) {
        glfwPollEvents();

        // mouse brush, in lattice coordinates
        int ww, wh, x0, y0, vw, vh;
        glfwGetWindowSize(win, &ww, &wh);
        letterbox(ww, wh, x0, y0, vw, vh);
        bool lb = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        bool rb = glfwGetMouseButton(win, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        if ((lb || rb) && vw > 0) {
            double mx, my;
            glfwGetCursorPos(win, &mx, &my);
            float lx = float((mx - x0) / vw * V.nx), ly = float((1 - (my - y0) / vh) * V.ny);
            float r = std::max(2.0f, V.ny / 60.0f);
            int N = V.nx * V.ny;
            paintKernel<<<(N + 255) / 256, 256>>>(V.g->p, V.g->dSolid, V.g->fA, lx, ly, r, lb);
            CUDA_CHECK(cudaGetLastError());
        }

        // simulate: moments are only needed after the last step of the frame
        int k = V.paused ? (V.stepOnce ? 1 : 0) : V.stepsPerFrame;
        V.stepOnce = false;
        for (int s = 0; s < k; s++) {
            V.g->setWriteMoments(s == k - 1);
            V.g->step();
        }
        V.step += k;
        stepsSince += k;

        if (++frame % 15 == 0) {  // drag/lift now and then (it waits for the GPU)
            F = V.g->wallForce();
            if (std::isnan(F.x)) { std::printf("diverged (NaN) at step %lld: resetting\n", V.step); reset(); continue; }
        }

        uploadField();

        int fw, fh;
        glfwGetFramebufferSize(win, &fw, &fh);
        letterbox(fw, fh, x0, y0, vw, vh);
        glViewport(0, 0, fw, fh);
        glClearColor(0.08f, 0.08f, 0.09f, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(x0, y0, vw, vh);
        glUseProgram(V.prog);
        glUniform1i(locMode, V.mode);
        glUniform1f(locScale, V.scale[V.mode]);
        glBindVertexArray(V.vao);
        glBindTexture(GL_TEXTURE_2D, V.tex);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glfwSwapBuffers(win);

        double now = glfwGetTime();
        if (now - tLast > 0.5) {
            double mlups = stepsSince * double(V.nx) * V.ny / (now - tLast) / 1e6;
            double q = 0.5 * V.U * V.U * V.D;
            char title[256];
            std::snprintf(title, sizeof title,
                          "LBM GPU  |  %s %dx%d  |  Re %.0f  |  step %lld  |  %.0f MLUPS, %d steps/frame  |  "
                          "C_D %.3f  C_L %+.3f  |  %s%s%s",
                          V.caseId ? "Schafer-Turek channel" : "cylinder", V.nx, V.ny, V.Re, V.step, mlups,
                          V.stepsPerFrame, F.x / q, F.y / q, V.mode ? "speed" : "vorticity",
                          V.res ? "  |  interop" : "", V.paused ? "  [PAUSED]" : "");
            glfwSetWindowTitle(win, title);
            tLast = now;
            stepsSince = 0;
        }
    }

    if (V.res) cudaGraphicsUnregisterResource(V.res);
    V.g.reset();
    glfwTerminate();
    return 0;
}
