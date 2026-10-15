// kmeans_serial.cpp
// Serial baseline implementation of Lloyd's K-means algorithm.
//
// This is the reference the parallel versions are validated against: given the
// same seed, dataset, and K, the MPI and CUDA versions must produce the same
// final inertia and (up to label permutation) the same assignments.
//
// Usage:
//   ./kmeans_serial <csv> <k> [--label-col C] [--max-iter N] [--tol T]
//                   [--seed S] [--out-labels F] [--out-centroids F]
#include "common.hpp"

#include <cstring>

using km::Dataset;

struct Args {
    std::string csv;
    int k = 10;
    int label_col = -1;
    int max_iter = 100;
    double tol = 1e-4;      // stop when fractional inertia change < tol
    unsigned seed = 1234;
    bool fixed_iter = false;  // if set, always run exactly max_iter (benchmarking)
    std::string out_labels;
    std::string out_centroids;
};

static Args parse(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr,
            "usage: %s <csv> <k> [--label-col C] [--max-iter N] [--tol T] "
            "[--seed S] [--out-labels F] [--out-centroids F]\n", argv[0]);
        std::exit(2);
    }
    Args a;
    a.csv = argv[1];
    a.k = std::atoi(argv[2]);
    for (int i = 3; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (s == "--label-col") a.label_col = std::atoi(next());
        else if (s == "--max-iter") a.max_iter = std::atoi(next());
        else if (s == "--tol") a.tol = std::atof(next());
        else if (s == "--seed") a.seed = static_cast<unsigned>(std::strtoul(next(), nullptr, 10));
        else if (s == "--fixed-iter") a.fixed_iter = true;
        else if (s == "--out-labels") a.out_labels = next();
        else if (s == "--out-centroids") a.out_centroids = next();
        else { std::fprintf(stderr, "unknown arg: %s\n", s.c_str()); std::exit(2); }
    }
    return a;
}

int main(int argc, char** argv) {
    Args args = parse(argc, argv);

    const double t_load0 = km::wtime();
    Dataset ds = km::read_csv(args.csv, args.label_col);
    const double t_load = km::wtime() - t_load0;

    const int n = ds.n, dims = ds.dims, k = args.k;
    std::printf("[serial] n=%d dims=%d k=%d  (load %.3fs)\n", n, dims, k, t_load);

    // Seed centroids with k-means++.
    std::vector<float> cents = km::kmeanspp_init(ds, k, args.seed);

    std::vector<int> assign(n, -1);
    std::vector<float> sums(static_cast<size_t>(k) * dims);
    std::vector<int> counts(k);

    double prev_inertia = -1.0;
    int iter = 0;
    const double t_cl0 = km::wtime();

    for (; iter < args.max_iter; ++iter) {
        // --- Assignment step: each point -> nearest centroid ---------------
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);
        double inertia = 0.0;
        int moved = 0;

        for (int i = 0; i < n; ++i) {
            const float* p = &ds.data[static_cast<size_t>(i) * dims];
            float best = std::numeric_limits<float>::max();
            int bestj = 0;
            for (int j = 0; j < k; ++j) {
                float d = km::dist2(p, &cents[static_cast<size_t>(j) * dims], dims);
                if (d < best) { best = d; bestj = j; }
            }
            if (assign[i] != bestj) { moved++; assign[i] = bestj; }
            inertia += best;
            // Accumulate into the running sum for the centroid update.
            float* srow = &sums[static_cast<size_t>(bestj) * dims];
            for (int d = 0; d < dims; ++d) srow[d] += p[d];
            counts[bestj]++;
        }

        // --- Update step: centroid = mean of assigned points ---------------
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                float inv = 1.0f / static_cast<float>(counts[j]);
                for (int d = 0; d < dims; ++d)
                    cents[static_cast<size_t>(j) * dims + d] =
                        sums[static_cast<size_t>(j) * dims + d] * inv;
            }
            // Empty clusters keep their previous position (simple, stable).
        }

        std::printf("  iter %3d  inertia %.6e  moved %d\n", iter, inertia, moved);

        // --- Convergence check ---------------------------------------------
        if (!args.fixed_iter && prev_inertia >= 0.0) {
            double rel = std::fabs(prev_inertia - inertia) /
                         (prev_inertia > 0 ? prev_inertia : 1.0);
            if (rel < args.tol || moved == 0) { prev_inertia = inertia; iter++; break; }
        }
        prev_inertia = inertia;
    }

    const double t_cluster = km::wtime() - t_cl0;
    std::printf("[serial] converged in %d iters  final_inertia %.6e  "
                "cluster_time %.3fs\n", iter, prev_inertia, t_cluster);
    std::printf("RESULT serial n=%d dims=%d k=%d iters=%d inertia=%.6e time=%.6f\n",
                n, dims, k, iter, prev_inertia, t_cluster);

    if (!args.out_labels.empty()) km::write_labels(args.out_labels, assign);
    if (!args.out_centroids.empty())
        km::write_centroids(args.out_centroids, cents, k, dims);
    return 0;
}
