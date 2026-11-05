// run_util.h — small helpers shared by the data-producing runners (sweep_*, run_*).
// Every CSV they write starts with '#' comment lines that record how it was made (PLAN.md §4:
// "every run writes its parameters ... and git hash into its output file").
// numpy.genfromtxt / pandas.read_csv(comment='#') skip those lines.

#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <string>

#ifdef _WIN32
#define LBM_POPEN _popen
#define LBM_PCLOSE _pclose
#else
#define LBM_POPEN popen
#define LBM_PCLOSE pclose
#endif

// short git hash of the checkout, plus "+dirty" if there are uncommitted changes
inline std::string gitHash() {
    std::string out;
    if (FILE* p = LBM_POPEN("git rev-parse --short HEAD 2>nul", "r")) {
        char buf[64] = {};
        if (std::fgets(buf, sizeof buf, p)) out = buf;
        LBM_PCLOSE(p);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    if (out.empty()) return "unknown";
    if (FILE* p = LBM_POPEN("git status --porcelain --untracked-files=no 2>nul", "r")) {
        char buf[8] = {};
        if (std::fgets(buf, sizeof buf, p)) out += "+dirty";
        LBM_PCLOSE(p);
    }
    return out;
}

inline std::string timestamp() {
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

// open a CSV in runs/. append = true keeps earlier rows (the header is written only if the file is
// new or empty), so a long campaign can be split over several invocations.
inline FILE* openCsv(const std::string& path, const char* header, const std::string& meta, bool append = false) {
    bool fresh = true;
    if (append)
        if (FILE* q = std::fopen(path.c_str(), "rb")) {
            std::fseek(q, 0, SEEK_END);
            fresh = std::ftell(q) == 0;
            std::fclose(q);
        }
    FILE* f = std::fopen(path.c_str(), append ? "a" : "w");
    if (!f) {
        std::fprintf(stderr, "cannot write %s (run from the repo root, and does runs/ exist?)\n", path.c_str());
        std::exit(1);
    }
    std::fprintf(f, "# %s  git %s  %s\n", path.c_str(), gitHash().c_str(), timestamp().c_str());
    if (!meta.empty()) std::fprintf(f, "# %s\n", meta.c_str());
    if (fresh) std::fprintf(f, "%s\n", header);
    std::fflush(f);
    return f;
}

struct Stopwatch {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double seconds() const { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};
