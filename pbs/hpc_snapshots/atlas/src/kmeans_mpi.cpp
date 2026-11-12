// kmeans_mpi.cpp
// Distributed-memory K-means using MPI (CPU only).
//
// Parallelization strategy (data parallelism over points):
//   1. Rank 0 loads the full dataset and k-means++ seeds the centroids.
//   2. Points are partitioned across ranks with MPI_Scatterv (near-equal
//      contiguous blocks). Each rank owns a disjoint slice and never needs
//      another rank's points.
//   3. Each iteration, every rank independently assigns its local points to
//      the nearest centroid and accumulates per-cluster partial sums + counts.
//   4. The partial sums/counts are combined with a single MPI_Allreduce so all
//      ranks hold the global sums -> every rank recomputes identical centroids.
//      This collective is the one unavoidable synchronization/communication
//      point per iteration and is the crux of the scaling analysis: its cost is
//      O(k*dims) and independent of n, so it amortizes well as n grows.
//
// Why Allreduce (not Reduce+Bcast): a single Allreduce is typically implemented
// as an efficient ring/recursive-doubling collective and leaves the reduced
// result on every rank in one step, avoiding a separate broadcast.
//
// Usage mirrors the serial version; run with mpirun -np P.
#include "common.hpp"

#include <mpi.h>
#include <cstring>

using km::Dataset;

struct Args {
    std::string csv;
    int k = 10;
    int label_col = -1;
    int max_iter = 100;
    double tol = 1e-4;
    unsigned seed = 1234;
    bool fixed_iter = false;  // always run exactly max_iter (benchmarking)
    std::string out_labels;
    std::string out_centroids;
};

static Args parse(int argc, char** argv) {
    Args a;
    if (argc < 3) {
        if (/*only rank 0 prints*/ true)
            std::fprintf(stderr, "usage: %s <csv> <k> [opts]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
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
    }
    return a;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    Args args = parse(argc, argv);

    // --- Rank 0 loads data; broadcast dims/n -------------------------------
    Dataset full;
    int n = 0, dims = 0;
    const int k = args.k;
    double t_load = 0.0;

    if (rank == 0) {
        double t0 = km::wtime();
        full = km::read_csv(args.csv, args.label_col);
        t_load = km::wtime() - t0;
        n = full.n;
        dims = full.dims;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dims, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- Compute per-rank point counts and scatter displacements -----------
    // Contiguous near-equal blocks: first (n % nproc) ranks get one extra point.
    std::vector<int> cnt(nproc), disp(nproc);
    {
        int base = n / nproc, rem = n % nproc, off = 0;
        for (int r = 0; r < nproc; ++r) {
            cnt[r] = base + (r < rem ? 1 : 0);
            disp[r] = off;
            off += cnt[r];
        }
    }
    const int local_n = cnt[rank];

    // Scatterv works in units of floats, so scale counts/displacements by dims.
    std::vector<int> fcnt(nproc), fdisp(nproc);
    for (int r = 0; r < nproc; ++r) { fcnt[r] = cnt[r] * dims; fdisp[r] = disp[r] * dims; }

    std::vector<float> local(static_cast<size_t>(local_n) * dims);
    MPI_Scatterv(rank == 0 ? full.data.data() : nullptr,
                 fcnt.data(), fdisp.data(), MPI_FLOAT,
                 local.data(), local_n * dims, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // --- Seed centroids on rank 0 (needs full data), broadcast -------------
    std::vector<float> cents(static_cast<size_t>(k) * dims);
    if (rank == 0) cents = km::kmeanspp_init(full, k, args.seed);
    MPI_Bcast(cents.data(), k * dims, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0)
        std::printf("[mpi] n=%d dims=%d k=%d nproc=%d  (load %.3fs)\n",
                    n, dims, k, nproc, t_load);

    // --- Iterate -----------------------------------------------------------
    std::vector<int> local_assign(local_n, -1);
    std::vector<float> local_sums(static_cast<size_t>(k) * dims);
    std::vector<float> global_sums(static_cast<size_t>(k) * dims);
    std::vector<long long> local_cnt(k), global_cnt(k);
    // Reduction phase per iteration = three MPI_Allreduce calls: the coordinate
    // sums (float), the per-cluster counts (long long), and a 2-element double
    // buffer carrying the scalar inertia + moved-point diagnostics. The counts
    // are kept separate from the sums because their datatypes differ.

    double prev_inertia = -1.0;
    int iter = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_cl0 = km::wtime();

    for (; iter < args.max_iter; ++iter) {
        std::fill(local_sums.begin(), local_sums.end(), 0.0f);
        std::fill(local_cnt.begin(), local_cnt.end(), 0LL);
        double local_inertia = 0.0;
        long long local_moved = 0;

        for (int i = 0; i < local_n; ++i) {
            const float* p = &local[static_cast<size_t>(i) * dims];
            float best = std::numeric_limits<float>::max();
            int bestj = 0;
            for (int j = 0; j < k; ++j) {
                float d = km::dist2(p, &cents[static_cast<size_t>(j) * dims], dims);
                if (d < best) { best = d; bestj = j; }
            }
            if (local_assign[i] != bestj) { local_moved++; local_assign[i] = bestj; }
            local_inertia += best;
            float* srow = &local_sums[static_cast<size_t>(bestj) * dims];
            for (int d = 0; d < dims; ++d) srow[d] += p[d];
            local_cnt[bestj]++;
        }

        // --- Global reduction of partial sums/counts -----------------------
        MPI_Allreduce(local_sums.data(), global_sums.data(), k * dims,
                      MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_cnt.data(), global_cnt.data(), k,
                      MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);

        double red[2] = {local_inertia, static_cast<double>(local_moved)};
        double gred[2];
        MPI_Allreduce(red, gred, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double inertia = gred[0];
        long long moved = static_cast<long long>(gred[1]);

        // Every rank recomputes centroids identically from global sums.
        for (int j = 0; j < k; ++j) {
            if (global_cnt[j] > 0) {
                float inv = 1.0f / static_cast<float>(global_cnt[j]);
                for (int d = 0; d < dims; ++d)
                    cents[static_cast<size_t>(j) * dims + d] =
                        global_sums[static_cast<size_t>(j) * dims + d] * inv;
            }
        }

        if (rank == 0)
            std::printf("  iter %3d  inertia %.6e  moved %lld\n", iter, inertia, moved);

        if (!args.fixed_iter && prev_inertia >= 0.0) {
            double rel = std::fabs(prev_inertia - inertia) /
                         (prev_inertia > 0 ? prev_inertia : 1.0);
            if (rel < args.tol || moved == 0) { prev_inertia = inertia; iter++; break; }
        }
        prev_inertia = inertia;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_cluster = km::wtime() - t_cl0;

    if (rank == 0) {
        std::printf("[mpi] converged in %d iters  final_inertia %.6e  "
                    "cluster_time %.3fs\n", iter, prev_inertia, t_cluster);
        std::printf("RESULT mpi n=%d dims=%d k=%d nproc=%d iters=%d "
                    "inertia=%.6e time=%.6f\n",
                    n, dims, k, nproc, iter, prev_inertia, t_cluster);
    }

    // --- Optional: gather assignments to rank 0 for validation -------------
    if (!args.out_labels.empty()) {
        std::vector<int> all_assign;
        if (rank == 0) all_assign.resize(n);
        MPI_Gatherv(local_assign.data(), local_n, MPI_INT,
                    rank == 0 ? all_assign.data() : nullptr,
                    cnt.data(), disp.data(), MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) km::write_labels(args.out_labels, all_assign);
    }
    if (rank == 0 && !args.out_centroids.empty())
        km::write_centroids(args.out_centroids, cents, k, dims);

    MPI_Finalize();
    return 0;
}
