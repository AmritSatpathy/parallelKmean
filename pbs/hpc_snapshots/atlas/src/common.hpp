// common.hpp
// Shared utilities for the K-means project: dataset I/O, timing, k-means++
// seeding, and the serial Lloyd update used by every implementation so that
// results are directly comparable.
//
// Data layout: all point/centroid coordinates are stored row-major in a flat
// std::vector<float>. Point i, dimension d lives at index i*dims + d. This flat
// layout is what the MPI scatter and the CUDA kernels expect, so keeping it
// everywhere avoids conversions.
#ifndef KMEANS_COMMON_HPP
#define KMEANS_COMMON_HPP

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace km {

// ---- Timing ---------------------------------------------------------------
// Simple monotonic wall-clock timer in seconds.
inline double wtime() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

// ---- Dataset --------------------------------------------------------------
struct Dataset {
    int n = 0;              // number of points
    int dims = 0;           // dimensionality
    std::vector<float> data;    // n*dims, row-major
    std::vector<int> labels;    // optional ground-truth labels (size n or 0)
};

// Read a numeric CSV. If `label_col` >= 0, that column is pulled out as an
// integer ground-truth label and excluded from the feature vector. A header
// row is auto-detected (first line contains any non-numeric token).
inline Dataset read_csv(const std::string& path, int label_col = -1) {
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
        std::exit(1);
    }

    Dataset ds;
    std::string line;
    bool first = true;
    int ncols = -1;

    while (std::getline(in, line)) {
        if (line.empty()) continue;

        // Split on commas (and whitespace, to be forgiving).
        std::vector<std::string> tok;
        std::string cur;
        for (char c : line) {
            if (c == ',' || c == '\t' || c == ' ' || c == '\r') {
                if (!cur.empty()) { tok.push_back(cur); cur.clear(); }
            } else {
                cur.push_back(c);
            }
        }
        if (!cur.empty()) tok.push_back(cur);
        if (tok.empty()) continue;

        // Header detection: if any token in the first row isn't a number, skip it.
        if (first) {
            first = false;
            bool is_header = false;
            for (const auto& t : tok) {
                char* end = nullptr;
                std::strtod(t.c_str(), &end);
                if (end == t.c_str() || *end != '\0') { is_header = true; break; }
            }
            ncols = static_cast<int>(tok.size());
            if (is_header) continue;  // consumed header, move to data rows
        }

        if (ncols < 0) ncols = static_cast<int>(tok.size());

        int fcols = (label_col >= 0) ? ncols - 1 : ncols;
        if (ds.dims == 0) ds.dims = fcols;

        for (int c = 0; c < static_cast<int>(tok.size()); ++c) {
            if (c == label_col) {
                ds.labels.push_back(static_cast<int>(std::strtol(tok[c].c_str(), nullptr, 10)));
            } else {
                ds.data.push_back(static_cast<float>(std::strtod(tok[c].c_str(), nullptr)));
            }
        }
        ds.n += 1;
    }

    if (ds.n == 0) {
        std::fprintf(stderr, "ERROR: no data rows parsed from %s\n", path.c_str());
        std::exit(1);
    }
    return ds;
}

// Write cluster assignments (one integer per line) so external tools can score.
inline void write_labels(const std::string& path, const std::vector<int>& a) {
    std::ofstream out(path);
    for (int v : a) out << v << "\n";
}

// Write centroids as CSV (k rows of `dims` floats).
inline void write_centroids(const std::string& path, const std::vector<float>& c,
                            int k, int dims) {
    std::ofstream out(path);
    out.setf(std::ios::fixed);
    out.precision(6);
    for (int j = 0; j < k; ++j) {
        for (int d = 0; d < dims; ++d) {
            out << c[j * dims + d];
            out << (d + 1 == dims ? '\n' : ',');
        }
    }
}

// ---- Distance -------------------------------------------------------------
// Squared Euclidean distance between point p (offset into data) and centroid c.
inline float dist2(const float* p, const float* c, int dims) {
    float s = 0.0f;
    for (int d = 0; d < dims; ++d) {
        float diff = p[d] - c[d];
        s += diff * diff;
    }
    return s;
}

// ---- Initialization -------------------------------------------------------
// k-means++ seeding. Deterministic given `seed` so that serial, MPI, and CUDA
// runs start from identical centroids -> identical results for validation.
// Runs on the full dataset; in MPI we seed on rank 0 then broadcast.
inline std::vector<float> kmeanspp_init(const Dataset& ds, int k, unsigned seed) {
    const int n = ds.n, dims = ds.dims;
    std::vector<float> cents(static_cast<size_t>(k) * dims);
    std::mt19937 rng(seed);

    // First centroid: uniform random point.
    std::uniform_int_distribution<int> pick(0, n - 1);
    int first = pick(rng);
    std::copy_n(&ds.data[static_cast<size_t>(first) * dims], dims, &cents[0]);

    std::vector<float> d2(n, std::numeric_limits<float>::max());
    for (int c = 1; c < k; ++c) {
        // Update nearest-centroid distance using the centroid just added.
        double total = 0.0;
        const float* newc = &cents[static_cast<size_t>(c - 1) * dims];
        for (int i = 0; i < n; ++i) {
            float dd = dist2(&ds.data[static_cast<size_t>(i) * dims], newc, dims);
            if (dd < d2[i]) d2[i] = dd;
            total += d2[i];
        }
        // Choose next centroid with probability proportional to D^2.
        std::uniform_real_distribution<double> U(0.0, total);
        double r = U(rng);
        int chosen = n - 1;
        for (int i = 0; i < n; ++i) {
            r -= d2[i];
            if (r <= 0.0) { chosen = i; break; }
        }
        std::copy_n(&ds.data[static_cast<size_t>(chosen) * dims], dims,
                    &cents[static_cast<size_t>(c) * dims]);
    }
    return cents;
}

// ---- Convergence ----------------------------------------------------------
// Objective: within-cluster sum of squared errors (inertia). Lower is better.
// We report this each iteration; it decreases monotonically for Lloyd's method,
// which is a cheap internal sanity check.
struct IterStats {
    double inertia = 0.0;
    int moved = 0;  // number of points that changed cluster this iteration
};

}  // namespace km

#endif  // KMEANS_COMMON_HPP
