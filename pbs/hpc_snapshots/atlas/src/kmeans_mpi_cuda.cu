// kmeans_mpi_cuda.cu
// Distributed K-means with MPI across ranks and CUDA within each rank.
//
// Hierarchy of parallelism:
//   - MPI (inter-node / inter-GPU): same point partitioning as kmeans_mpi.cpp.
//     Each rank owns a disjoint slice of points and one GPU.
//   - CUDA (intra-rank): the assignment step -- the O(n*k*dims) hot loop -- is
//     offloaded to the GPU. One thread per point finds its nearest centroid and
//     atomically accumulates that point's coordinates into per-cluster sums.
//
// Per-iteration flow on each rank:
//   1. Copy current centroids host->device (small: k*dims floats).
//   2. Launch assign kernel: computes assignments, device partial sums/counts,
//      and a per-point inertia contribution (block-reduced).
//   3. Copy device partial sums/counts/inertia back to host (small).
//   4. MPI_Allreduce the partial sums/counts/inertia across ranks (identical to
//      the pure-MPI version), then recompute centroids on host.
//
// Design decisions worth calling out in the report:
//   * Centroids live in global memory but are cached through __ldg/read-only;
//     for modest k*dims they also fit in shared memory, which we exploit.
//   * Accumulation uses atomicAdd into global sums. This is the GPU analogue of
//     the serial reduction. Contention is bounded by k (number of clusters),
//     not n, so it stays cheap for small/moderate k.
//   * Points are copied to the device once (they don't change); only centroids
//     move each iteration. This keeps PCIe traffic O(k*dims) per iteration.
//   * GPU-per-rank mapping uses rank % deviceCount so multiple ranks can share
//     a node's GPUs when oversubscribed.
#include "common.hpp"

#include <mpi.h>
#include <cuda_runtime.h>
#include <cstring>

using km::Dataset;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(_e));                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                      \
        }                                                                      \
    } while (0)

// Assignment kernel. Shared memory holds the centroids when they fit; one
// thread handles one point. Partial sums and counts are accumulated with
// atomicAdd into global arrays. Each thread also adds its point's squared
// distance to the chosen centroid directly into a global inertia accumulator
// via atomicAdd on a double (one atomic per point; not block-reduced). This is
// simple and the inertia atomics are a small share of the kernel's O(n*k*dims)
// distance work; a per-block reduction would reduce contention if needed.
__global__ void assign_kernel(const float* __restrict__ points, int local_n,
                              int dims, const float* __restrict__ cents, int k,
                              float* sums, int* counts, int* assign,
                              double* inertia, bool use_shared) {
    extern __shared__ float s_cents[];  // k*dims floats when launched with shmem

    // Cooperatively load centroids into shared memory (only when the launch
    // allocated it; otherwise read centroids straight from global memory so we
    // never touch unallocated shared memory).
    if (use_shared) {
        const int tsz = blockDim.x;
        for (int idx = threadIdx.x; idx < k * dims; idx += tsz)
            s_cents[idx] = cents[idx];
        __syncthreads();
    }
    const float* cbase = use_shared ? s_cents : cents;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= local_n) return;

    const float* p = points + static_cast<size_t>(i) * dims;
    float best = 3.4e38f;
    int bestj = 0;
    for (int j = 0; j < k; ++j) {
        const float* c = cbase + j * dims;
        float d = 0.0f;
        for (int dd = 0; dd < dims; ++dd) {
            float diff = p[dd] - c[dd];
            d += diff * diff;
        }
        if (d < best) { best = d; bestj = j; }
    }

    assign[i] = bestj;
    atomicAdd(inertia, static_cast<double>(best));
    atomicAdd(&counts[bestj], 1);
    float* srow = sums + static_cast<size_t>(bestj) * dims;
    for (int dd = 0; dd < dims; ++dd)
        atomicAdd(&srow[dd], p[dd]);
}

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

    // --- Bind this rank to a GPU -------------------------------------------
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        if (rank == 0) std::fprintf(stderr, "ERROR: no CUDA devices visible\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % ndev));

    // --- Load + broadcast dims/n -------------------------------------------
    Dataset full;
    int n = 0, dims = 0;
    const int k = args.k;
    double t_load = 0.0;
    if (rank == 0) {
        double t0 = km::wtime();
        full = km::read_csv(args.csv, args.label_col);
        t_load = km::wtime() - t0;
        n = full.n; dims = full.dims;
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dims, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // --- Partition + scatter points ----------------------------------------
    std::vector<int> cnt(nproc), disp(nproc);
    {
        int base = n / nproc, rem = n % nproc, off = 0;
        for (int r = 0; r < nproc; ++r) {
            cnt[r] = base + (r < rem ? 1 : 0);
            disp[r] = off; off += cnt[r];
        }
    }
    const int local_n = cnt[rank];
    std::vector<int> fcnt(nproc), fdisp(nproc);
    for (int r = 0; r < nproc; ++r) { fcnt[r] = cnt[r] * dims; fdisp[r] = disp[r] * dims; }

    std::vector<float> local(static_cast<size_t>(local_n) * dims);
    MPI_Scatterv(rank == 0 ? full.data.data() : nullptr,
                 fcnt.data(), fdisp.data(), MPI_FLOAT,
                 local.data(), local_n * dims, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // --- Seed + broadcast centroids ----------------------------------------
    std::vector<float> cents(static_cast<size_t>(k) * dims);
    if (rank == 0) cents = km::kmeanspp_init(full, k, args.seed);
    MPI_Bcast(cents.data(), k * dims, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0)
        std::printf("[mpi+cuda] n=%d dims=%d k=%d nproc=%d gpus/node~%d (load %.3fs)\n",
                    n, dims, k, nproc, ndev, t_load);

    // --- Device allocations (points copied once) ---------------------------
    float *d_points = nullptr, *d_cents = nullptr, *d_sums = nullptr;
    int *d_counts = nullptr, *d_assign = nullptr;
    double *d_inertia = nullptr;
    size_t pbytes = static_cast<size_t>(local_n) * dims * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_points, pbytes ? pbytes : 1));
    CUDA_CHECK(cudaMalloc(&d_cents, static_cast<size_t>(k) * dims * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums, static_cast<size_t>(k) * dims * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counts, static_cast<size_t>(k) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_assign, static_cast<size_t>(local_n ? local_n : 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_inertia, sizeof(double)));
    if (pbytes) CUDA_CHECK(cudaMemcpy(d_points, local.data(), pbytes, cudaMemcpyHostToDevice));

    const int TPB = 256;
    const int blocks = (local_n + TPB - 1) / TPB;
    const size_t shmem = static_cast<size_t>(k) * dims * sizeof(float);

    // If centroids don't fit in shared memory, fall back to a 0-shmem launch
    // that reads centroids from global memory (kernel still works: when shmem
    // is too large we cap it and read globally). Keep it simple: require fit.
    int maxShmem = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxShmem,
               cudaDevAttrMaxSharedMemoryPerBlock, rank % ndev));
    bool use_shared = shmem <= static_cast<size_t>(maxShmem);

    std::vector<float> h_sums(static_cast<size_t>(k) * dims), g_sums(static_cast<size_t>(k) * dims);
    std::vector<int> h_counts(k);
    std::vector<long long> g_cnt(k);

    double prev_inertia = -1.0;
    int iter = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_cl0 = km::wtime();

    for (; iter < args.max_iter; ++iter) {
        CUDA_CHECK(cudaMemcpy(d_cents, cents.data(),
                   static_cast<size_t>(k) * dims * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_sums, 0, static_cast<size_t>(k) * dims * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_counts, 0, static_cast<size_t>(k) * sizeof(int)));
        CUDA_CHECK(cudaMemset(d_inertia, 0, sizeof(double)));

        if (local_n > 0) {
            assign_kernel<<<blocks, TPB, use_shared ? shmem : 0>>>(
                d_points, local_n, dims, d_cents, k,
                d_sums, d_counts, d_assign, d_inertia, use_shared);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy device partial results back.
        CUDA_CHECK(cudaMemcpy(h_sums.data(), d_sums,
                   static_cast<size_t>(k) * dims * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_counts.data(), d_counts,
                   static_cast<size_t>(k) * sizeof(int), cudaMemcpyDeviceToHost));
        double local_inertia = 0.0;
        CUDA_CHECK(cudaMemcpy(&local_inertia, d_inertia, sizeof(double), cudaMemcpyDeviceToHost));

        std::vector<long long> local_cnt(k);
        for (int j = 0; j < k; ++j) local_cnt[j] = h_counts[j];

        // --- Global reduction across ranks (same as pure MPI) --------------
        MPI_Allreduce(h_sums.data(), g_sums.data(), k * dims,
                      MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(local_cnt.data(), g_cnt.data(), k,
                      MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        double g_inertia = 0.0;
        MPI_Allreduce(&local_inertia, &g_inertia, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for (int j = 0; j < k; ++j) {
            if (g_cnt[j] > 0) {
                float inv = 1.0f / static_cast<float>(g_cnt[j]);
                for (int d = 0; d < dims; ++d)
                    cents[static_cast<size_t>(j) * dims + d] =
                        g_sums[static_cast<size_t>(j) * dims + d] * inv;
            }
        }

        if (rank == 0)
            std::printf("  iter %3d  inertia %.6e\n", iter, g_inertia);

        if (!args.fixed_iter && prev_inertia >= 0.0) {
            double rel = std::fabs(prev_inertia - g_inertia) /
                         (prev_inertia > 0 ? prev_inertia : 1.0);
            if (rel < args.tol) { prev_inertia = g_inertia; iter++; break; }
        }
        prev_inertia = g_inertia;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t_cluster = km::wtime() - t_cl0;

    if (rank == 0) {
        std::printf("[mpi+cuda] converged in %d iters  final_inertia %.6e  "
                    "cluster_time %.3fs\n", iter, prev_inertia, t_cluster);
        std::printf("RESULT mpi_cuda n=%d dims=%d k=%d nproc=%d iters=%d "
                    "inertia=%.6e time=%.6f\n",
                    n, dims, k, nproc, iter, prev_inertia, t_cluster);
    }

    // --- Optional: gather assignments for validation -----------------------
    if (!args.out_labels.empty()) {
        std::vector<int> local_assign(local_n);
        if (local_n > 0)
            CUDA_CHECK(cudaMemcpy(local_assign.data(), d_assign,
                       static_cast<size_t>(local_n) * sizeof(int), cudaMemcpyDeviceToHost));
        std::vector<int> all_assign;
        if (rank == 0) all_assign.resize(n);
        MPI_Gatherv(local_assign.data(), local_n, MPI_INT,
                    rank == 0 ? all_assign.data() : nullptr,
                    cnt.data(), disp.data(), MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) km::write_labels(args.out_labels, all_assign);
    }
    if (rank == 0 && !args.out_centroids.empty())
        km::write_centroids(args.out_centroids, cents, k, dims);

    cudaFree(d_points); cudaFree(d_cents); cudaFree(d_sums);
    cudaFree(d_counts); cudaFree(d_assign); cudaFree(d_inertia);
    MPI_Finalize();
    return 0;
}
