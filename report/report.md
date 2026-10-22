---
title: "Parallelising K-means Clustering with MPI and CUDA"
subtitle: "Scalable Distributed Computing for Data Science DSA5208"
author: "Amrit Satpathy (A0355181E) · Ritik Srivastava (A0355860X)"
date: "October 2026"
geometry: margin=1in
---

<!--
Export to PDF with either:
  pandoc report/report.md -o report/report.pdf
  (needs pandoc + a LaTeX engine, e.g. texlive-xetex)
or open in a Markdown editor and "export to PDF".
Figures referenced below live in ../results/. Regenerate the data charts with
  scripts/plot_scaling.py        (speedup/efficiency/runtime from a scaling CSV)
  scripts/plot_illustrations.py  (cross-platform, comm-vs-compute, per-iteration)
The conceptual diagrams are hand-authored SVGs committed alongside:
  diagram_lloyd.svg, diagram_mpi_dataflow.svg, diagram_cuda_memory.svg
Section 7 holds the local reference numbers; Section 8
holds the required Atlas/Vanda runtime figures.
-->

## Required deliverables — where each is covered

| Required item | Section |
|---------------|---------|
| Introduction to the algorithm | 1 |
| Design decisions in the parallelisation | 2, 4.3, 5.3 (and 3–5 overall) |
| Parallel scaling and efficiency | 7 (local), 8 (cluster) |
| Analysis and discussion | 9 |
| Validation that the parallel algorithm is correct | 6 |
| Runtime figures from Atlas or Vanda | 8 |
| Conclusions about the real data | 10 (MNIST) |
| Conclusion | 12 |
| Explanation of AI usage | 13 |
| References | 14 |

(Section 9 is analysis and discussion; Section 11 documents limitations of the
cluster experiments.)

## 1. Introduction: what is K-means?

### 1.1 The problem

K-means is one of the most widely used *unsupervised* clustering algorithms. Given
a set of `n` data points in `d`-dimensional space and a target number of clusters
`K`, it partitions the points into `K` groups so that points within a group are
close together and points in different groups are far apart. "Close" is measured
by Euclidean distance.

Formally, K-means chooses `K` cluster centres (called **centroids**)
`μ_1, …, μ_K` and an assignment `a(i)` of each point `x_i` to a centroid, so as to
minimise the **within-cluster sum of squared distances**, also called the
*inertia*:

$$ J \;=\; \sum_{i=1}^{n} \bigl\lVert x_i - \mu_{a(i)} \bigr\rVert^2 . $$

Finding the global optimum is NP-hard, so in practice we use an iterative
heuristic that is guaranteed to decrease `J` every step until it settles into a
local optimum.

### 1.2 Lloyd's algorithm

The standard heuristic is **Lloyd's algorithm**. After choosing initial
centroids, it repeats two steps until the assignments stop changing (or the
inertia stops decreasing by more than a small tolerance):

1. **Assignment step.** Assign every point to the nearest centroid:
   $$ a(i) = \arg\min_{j}\; \lVert x_i - \mu_j \rVert^2 . $$
   For each of the `n` points we compute the distance to each of the `K`
   centroids, and each distance touches `d` coordinates. Cost: **`O(n·K·d)`**.

2. **Update step.** Move each centroid to the mean of the points currently
   assigned to it:
   $$ \mu_j = \frac{1}{|C_j|}\sum_{i \in C_j} x_i . $$
   Each point contributes to exactly one centroid's sum, so this is a
   **reduction** (a sum) over the data. Cost: **`O(n·d)`**.

Because the assignment step is `K` times more expensive than the update step, it
is the computational hot spot and the obvious target for parallelisation.

![Lloyd's algorithm: the assignment step (orange) is the `O(n·K·d)` hot spot that dominates each iteration; the update step and seeding are cheap `O(n·d)`/`O(K·d)` work.](../results/diagram_lloyd.svg){ width=42% }

A small practical detail: squared Euclidean distance is used throughout. The
square root is unnecessary because it does not change which centroid is closest,
and avoiding it saves work in the inner loop.

### 1.3 Initialisation: k-means++

Lloyd's algorithm only finds a local optimum, and a poor set of starting
centroids can give a poor result. We seed centroids with **k-means++**, which
picks the first centroid at random and then picks each subsequent centroid with
probability proportional to its squared distance from the nearest already-chosen
centroid. This spreads the initial centroids out and gives better, more
repeatable results than uniform random seeding. We seed deterministically from a
fixed `--seed`, so every version of the program (serial, MPI, CUDA) starts from
*identical* centroids — which is what lets us validate the parallel versions
against the serial baseline (Section 6).

### 1.4 Why K-means is a good parallelisation target

In the vocabulary of parallel computing, K-means is naturally a **data-parallel**
problem: the same operation (find the nearest centroid) is applied to many
independent pieces of data (the points). Data parallelism is attractive because
it has little synchronisation delay and avoids race conditions, at the cost of
being less flexible than task parallelism — a trade-off that suits K-means well.

The assignment step is **embarrassingly parallel**: each point's nearest-centroid
search is independent of every other point. If that were the whole algorithm, a
parallel version would be trivial and uninteresting.

The update step is what makes it interesting. A centroid's new position depends
on *all* the points assigned to it. If we spread the points across many
processes, no single process has all the points for any cluster, so the processes
must **communicate** to combine their partial results, and they must
**synchronise** once per iteration before anyone can take the next step. This
communication-and-synchronisation pattern — a global **reduction** every
iteration — is exactly the non-trivial behaviour this project sets out to study.

This also sets the ceiling on achievable speedup. By **Amdahl's law**, the
maximum strong-scaling speedup is `1 / (S + P/n)`, where `S` is the serial
fraction; here the only inherently non-parallel work per iteration is the small
`O(K·d)` reduction and the synchronisation around it, so `S` is tiny and the
algorithm can in principle scale to large process counts — which the measured
results in Sections 7–8 confirm.

## 2. How we plan to parallelise it

Before writing any parallel code, we settled on the strategy below. The guiding
idea is **data parallelism over points**: split the `n` points across workers,
let each worker do the independent per-point work locally, and combine results
only where the algorithm genuinely requires it (the centroid update).

### 2.1 Decomposition

We partition the `n` points into near-equal contiguous blocks, one block per
worker. Each worker is responsible only for its own points and never needs
another worker's raw points. The centroids, by contrast, are **small**
(`K·d` numbers) and are needed in full by every worker, so every worker keeps a
complete copy of the centroids.

This choice is deliberate:

- The per-point work (assignment) is uniform, so equal-size blocks give balanced
  load without any dynamic scheduling.
- Points are the big data (`n·d`); centroids are tiny (`K·d`). Partitioning the
  big thing and replicating the small thing minimises both memory and
  communication.

### 2.2 What has to be communicated, and how often

Each iteration, every worker can independently:

- assign its local points to the nearest centroid, and
- accumulate, for each cluster, the **partial sum** of its local points and a
  **partial count** of how many it assigned.

To finish the update step, the partial sums and partial counts from all workers
must be added together. The crucial observation is that we communicate the
**partial sums**, not the assignments. The partial sums are of size `K·d`
(independent of `n`), whereas the assignments would be of size `n`. For our large
run, `K·d = 128·64 = 8192` numbers versus `n = 3,000,000` — three orders of
magnitude less traffic. This single design decision is why the method scales
well.

So the per-iteration pattern is:

```
every worker:  assign local points  ->  local partial sums + counts
all workers:   SUM-reduce the partial sums + counts   (global reduction phase)
every worker:  centroid_j = global_sum_j / global_count_j
```

The global reduction is the one communication-and-synchronisation *phase* per
iteration. In our implementation it is realised as a small, fixed number of
`MPI_Allreduce` collectives (sum the coordinate sums, sum the counts, and sum the
scalar inertia/moved diagnostics — three in total; see Section 4.3), each of size
`O(K·d)` or smaller. Its cost depends on `K·d` and the number of workers, but not
on `n`, so as the dataset grows the reduction is amortised over more and more
local compute.

![Per-iteration dataflow of the MPI design: points are scattered once, each rank assigns its own block independently (green), and an `MPI_Allreduce` reduction phase (red) combines the small `O(K·d)` partial sums — the only synchronisation per iteration. (The phase is three `Allreduce` calls in the code: sums, counts, and the scalar diagnostics.)](../results/diagram_mpi_dataflow.svg){ width=90% }

The imbalance this exploits is stark. Each iteration the assignment step performs
`O(n·K·d)` work while the reduction moves only `K·d = 8192` floats, a ratio of
millions-to-one for our dataset. This is the whole reason the method scales, and
it is worth seeing to scale:

![Per-iteration assignment compute (`O(n·K·d)`) dwarfs the reduction traffic (`O(K·d)`) by about six orders of magnitude; the reduction volume does not grow with `n`.](../results/comm_vs_compute.png){ width=55% }

### 2.3 Two levels of parallelism

We implement this plan at two levels, which map onto the two classical models of
process interaction (message passing between separate memory pools, and shared
memory within a device) and can be combined as the **MPI + X** pattern:

- **MPI (between processes / nodes) — message passing.** Processes ("ranks")
  each own a block of points in their own private memory and exchange data only
  through explicit messages. The global reduction uses the `MPI_Allreduce`
  collective. This is the standard way to parallelise across separate memory
  pools and scales across the cores of a machine and across nodes of a cluster.
  In Flynn's taxonomy this is a **MIMD** system: each rank runs its own
  instruction stream over its own data.

- **CUDA (within a process, on a GPU) — shared memory.** The assignment step for
  one rank's block is offloaded to a GPU, with one GPU thread per point. A GPU is
  a **SIMD**-style accelerator: it is built for exactly this kind of regular
  floating-point arithmetic over large arrays, where many threads apply the same
  instruction to different data. The GPU performs the local partial-sum
  accumulation in its shared/global memory; the cross-rank combination is still
  the same `MPI_Allreduce`. This targets the hot `O(n·K·d)` loop with thousands
  of hardware threads.

This is the **MPI + CUDA** structure recommended for the project: MPI handles
inter-node parallelism, CUDA handles the intra-node (intra-GPU) compute, and the
communication volume between ranks is unchanged from the pure-MPI version. The
three programs we built are therefore: a **serial** baseline, an **MPI** (CPU)
version, and an **MPI+CUDA** (GPU) version that nests CUDA inside the MPI
decomposition.

## 3. Serial baseline

The serial program is a direct transcription of Lloyd's algorithm and serves as
the correctness reference. Its core is the two-step loop. The assignment step
finds the nearest centroid for each point and, in the same pass, accumulates the
running sums used by the update step:

```cpp
for (int i = 0; i < n; ++i) {
    const float* p = &ds.data[i * dims];
    float best = std::numeric_limits<float>::max();
    int bestj = 0;
    for (int j = 0; j < k; ++j) {                 // nearest-centroid search
        float d = km::dist2(p, &cents[j * dims], dims);
        if (d < best) { best = d; bestj = j; }
    }
    assign[i] = bestj;
    inertia += best;
    float* srow = &sums[bestj * dims];            // accumulate for the update
    for (int d = 0; d < dims; ++d) srow[d] += p[d];
    counts[bestj]++;
}

// Update step: each centroid becomes the mean of its assigned points.
for (int j = 0; j < k; ++j)
    if (counts[j] > 0)
        for (int d = 0; d < dims; ++d)
            cents[j * dims + d] = sums[j * dims + d] / counts[j];
```

Everything below parallelises exactly this computation while keeping the result
identical.

## 4. MPI implementation (CPU)

### 4.1 Distributing the data

Rank 0 reads the dataset and seeds the centroids. The points are then handed out
with `MPI_Scatterv` (near-equal contiguous blocks), and the seeded centroids are
broadcast so every rank starts identically:

```cpp
// Near-equal contiguous blocks: first (n % nproc) ranks get one extra point.
int base = n / nproc, rem = n % nproc, off = 0;
for (int r = 0; r < nproc; ++r) {
    cnt[r]  = base + (r < rem ? 1 : 0);
    disp[r] = off;  off += cnt[r];
}
// Scatter the points (counts/displacements are in units of floats = dims each).
MPI_Scatterv(rank == 0 ? full.data.data() : nullptr,
             fcnt.data(), fdisp.data(), MPI_FLOAT,
             local.data(), local_n * dims, MPI_FLOAT, 0, MPI_COMM_WORLD);

// Everyone needs the same starting centroids.
if (rank == 0) cents = km::kmeanspp_init(full, k, args.seed);
MPI_Bcast(cents.data(), k * dims, MPI_FLOAT, 0, MPI_COMM_WORLD);
```

### 4.2 The parallel iteration

Each rank runs the identical assignment loop as the serial version, but only over
its **local** points, producing local partial sums and counts. The only new code
is the global combination:

```cpp
// Local assignment over this rank's points -> local_sums, local_cnt, inertia.
for (int i = 0; i < local_n; ++i) {
    /* ... identical nearest-centroid search as serial ... */
    float* srow = &local_sums[bestj * dims];
    for (int d = 0; d < dims; ++d) srow[d] += p[d];
    local_cnt[bestj]++;
}

// The global reduction phase: one synchronisation point, realised as three
// MPI_Allreduce collectives (coordinate sums, counts, scalar diagnostics).
MPI_Allreduce(local_sums.data(), global_sums.data(), k * dims,
              MPI_FLOAT,     MPI_SUM, MPI_COMM_WORLD);
MPI_Allreduce(local_cnt.data(),  global_cnt.data(),  k,
              MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
double red[2]  = {local_inertia, (double)local_moved}, gred[2];
MPI_Allreduce(red, gred, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

// Every rank now holds the global sums and recomputes identical centroids.
for (int j = 0; j < k; ++j)
    if (global_cnt[j] > 0) {
        float inv = 1.0f / (float)global_cnt[j];
        for (int d = 0; d < dims; ++d)
            cents[j * dims + d] = global_sums[j * dims + d] * inv;
    }
```

### 4.3 Design decisions

Every step uses an MPI **collective communication** rather than hand-written
point-to-point `MPI_Send`/`MPI_Recv`, because collectives are implemented with
efficient communication patterns (e.g. a binary-tree reduction, which turns an
`O(n)` serial combine into `O(log n)` across ranks) and are less error-prone than
orchestrating sends and receives by hand.

- **`MPI_Bcast` after serial initialisation.** The k-means++ seeding is done once
  on rank 0 and broadcast — a textbook use of broadcast (distribute data produced
  by a serial init step), and more efficient than many individual sends.
- **`MPI_Scatterv` to distribute points.** The `v` variant lets the per-rank
  block sizes differ by one point when `n` is not divisible by the rank count.
- **`MPI_Allreduce`, not `MPI_Reduce` + `MPI_Bcast`.** A reduction combines data
  from every rank into one result; the *All* variant additionally leaves that
  result on *every* rank, so each rank can recompute the centroids itself. This
  avoids a separate broadcast and avoids a privileged "master" rank, keeping the
  ranks symmetric. The reduction phase is three `Allreduce` calls — the
  coordinate sums (`k·d` floats), the per-cluster counts (`k` longs), and a
  two-element scalar buffer carrying the summed inertia and moved-point count for
  the iteration log. The counts could be packed with the sums to save a call, but
  the datatypes differ (`float` vs `long long`) and the scalars are only
  diagnostics, so we kept them separate for clarity; all three are `O(K·d)` or
  smaller and independent of `n`.
- **Reduce sums, not assignments.** The messages are `O(K·d)`, independent of
  `n`, as planned in Section 2.2 — so communication volume does not grow with the
  dataset.
- **Contiguous block partition.** Simple, cache-friendly, and naturally balanced
  because per-point work is uniform (no dynamic scheduling needed).
- **One reduction phase's worth of synchronisation per iteration.** The three
  back-to-back `Allreduce` calls form a single unavoidable barrier; everything
  else is embarrassingly parallel. We use the default blocking collectives: the
  risk with blocking communication is ranks idling at the barrier or deadlock,
  but with symmetric `Allreduce`s per iteration and balanced load, neither is a
  problem here.

## 5. MPI + CUDA implementation (GPU)

The GPU version keeps the exact MPI structure of Section 4 — scatter points,
run the `Allreduce` reduction phase on the partial sums, recompute centroids —
but replaces each rank's local assignment loop with a CUDA kernel. One MPI rank
is bound to one GPU (`cudaSetDevice(rank % deviceCount)`); this maps one rank to
one device provided the job places at most as many ranks on a node as it has
GPUs, which the PBS scripts ensure by setting `mpiprocs = ngpus`.

### 5.1 The assignment kernel

One GPU thread handles one point. Centroids are staged in fast **shared memory**
(they are small and read by every thread), each thread finds its nearest centroid,
and then accumulates its contribution into the per-cluster sums with `atomicAdd` —
the GPU equivalent of the serial running sum:

```cpp
__global__ void assign_kernel(const float* points, int local_n, int dims,
                              const float* cents, int k,
                              float* sums, int* counts, int* assign,
                              double* inertia) {
    extern __shared__ float s_cents[];                 // centroids in shared mem
    for (int idx = threadIdx.x; idx < k * dims; idx += blockDim.x)
        s_cents[idx] = cents[idx];
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;     // one thread per point
    if (i >= local_n) return;

    const float* p = points + i * dims;
    float best = 3.4e38f; int bestj = 0;
    for (int j = 0; j < k; ++j) {                      // nearest-centroid search
        const float* c = s_cents + j * dims;
        float d = 0.0f;
        for (int dd = 0; dd < dims; ++dd) { float t = p[dd] - c[dd]; d += t * t; }
        if (d < best) { best = d; bestj = j; }
    }

    assign[i] = bestj;
    atomicAdd(inertia, (double)best);                  // reduction via atomics
    atomicAdd(&counts[bestj], 1);
    float* srow = sums + bestj * dims;
    for (int dd = 0; dd < dims; ++dd) atomicAdd(&srow[dd], p[dd]);
}
```

### 5.2 The host-side iteration

The points are copied to the device **once** before the loop (they never change).
Each iteration resets the device accumulators, uploads the small centroid array,
launches the kernel, copies the small partial results back, and feeds them into
the *same* reduction phase as the CPU version:

```cpp
for (iter = 0; iter < max_iter; ++iter) {
    cudaMemcpy(d_cents, cents.data(), k*dims*sizeof(float), H2D);  // small upload
    cudaMemset(d_sums,    0, k*dims*sizeof(float));   // reset accumulators
    cudaMemset(d_counts,  0, k*sizeof(int));          //   every iteration
    cudaMemset(d_inertia, 0, sizeof(double));         //   (no carry-over)

    assign_kernel<<<blocks, TPB, shmem>>>(d_points, local_n, dims, d_cents, k,
                                          d_sums, d_counts, d_assign, d_inertia);
    cudaGetLastError();            // catch launch errors
    cudaDeviceSynchronize();       // kernel done before we read the results back

    cudaMemcpy(h_sums.data(),   d_sums,   k*dims*sizeof(float), D2H);
    cudaMemcpy(h_counts.data(), d_counts, k*sizeof(int),        D2H);  // int on device
    cudaMemcpy(&local_inertia,  d_inertia, sizeof(double),      D2H);
    // Promote int device counts to long long to match the MPI datatype below.
    for (int j = 0; j < k; ++j) local_cnt[j] = h_counts[j];

    // Same three-call reduction phase as the pure-MPI version.
    MPI_Allreduce(h_sums.data(),  g_sums.data(), k*dims, MPI_FLOAT,     MPI_SUM, comm);
    MPI_Allreduce(local_cnt.data(), g_cnt.data(), k,     MPI_LONG_LONG, MPI_SUM, comm);
    MPI_Allreduce(&local_inertia, &g_inertia,    1,      MPI_DOUBLE,    MPI_SUM, comm);
    /* recompute centroids from g_sums / g_cnt, exactly as before */
}
```

Note the count datatypes: the kernel accumulates counts into a device `int`
array (`atomicAdd` on `int` is fast), which is copied to a host `int` buffer and
then promoted to `long long` before the `MPI_LONG_LONG` reduction, so the MPI
datatype matches the reduced buffer. The device accumulators (`d_sums`,
`d_counts`, `d_inertia`) are zeroed at the top of every iteration, so no values
carry across iterations, and `cudaDeviceSynchronize()` guarantees the kernel has
finished before the partial results are copied back and reduced.

![MPI + CUDA memory and data movement. The big `O(n·d)` point array crosses the PCIe link exactly once (thick green); every iteration moves only the small `O(K·d)` centroids down and partial sums back (blue dashed). Inside the kernel, centroids are staged in shared memory and results accumulated with `atomicAdd`.](../results/diagram_cuda_memory.svg){ width=95% }

### 5.3 Design decisions

GPU performance is usually limited by memory movement rather than arithmetic, so
most of these decisions are about memory, not compute.

- **Points uploaded once.** The host↔device PCIe link is one of the slow
  communication paths in the machine, so moving data across it is minimised: the
  points are constant across iterations and copied to the device a single time.
  Per-iteration traffic is then only the `O(K·d)` centroid upload and the
  `O(K·d)` partial-sum download — never the `O(n·d)` dataset.
- **Centroids in shared memory.** Shared memory lives physically close to the GPU
  cores and is far faster than global memory. Every thread reads every centroid,
  so staging the centroids in `__shared__` memory (with one cooperative load per
  block, then a `__syncthreads()`) turns `K` repeated global reads per thread
  into one shared-memory read — the same tiling idea used for cache reuse on the
  CPU.
- **Coalesced access to the points.** Points are stored row-major and the hot
  reads run over a point's contiguous coordinates, so threads in a warp read
  neighbouring addresses; this lets the hardware coalesce accesses and use the
  full memory-bus width.
- **Little warp divergence.** The kernel's only branch is the nearest-centroid
  `if (d < best)`, a simple min-tracking update rather than a data-dependent
  control split, so threads in a warp follow essentially the same path and the
  SIMD units stay busy.
- **Accumulation via `atomicAdd`.** This is the GPU form of the serial reduction:
  each point adds its coordinates into its chosen cluster's sum, its count, and
  the global inertia. Atomics serialise concurrent updates to the same address,
  so they are a potential bottleneck. The number of *distinct* targets is bounded
  by `K` (points land in only `K` coordinate-sum rows and `K` counts), but that
  does **not** by itself make atomics cheap: a small `K` means many threads
  contend for the *same* few addresses, which can actually increase collisions,
  and every point also contends on the single global inertia accumulator. What
  keeps it acceptable here is that the atomic traffic is still `O(n)` additions
  spread over the kernel while the distance arithmetic is `O(n·K·d)`, so for our
  `K=128, d=64` the atomics are a small share of kernel time — but we did not
  isolate their cost with a profiler, so we treat this as a reasonable-cost
  choice rather than a measured one. A two-stage reduction (accumulate per block
  in shared memory, then one atomic per block per cluster) would cut contention
  and is the standard next optimisation; we did not need it to meet the project's
  goals and did not benchmark it.
- **One rank per GPU, same `MPI_Allreduce` as the CPU path.** The rank↔GPU
  mapping (`cudaSetDevice(rank % deviceCount)`) means no explicit device
  switching and no increase in inter-rank communication volume versus the
  CPU version — the recommended MPI+CUDA structure. The GPU only changes *how* a
  rank computes its local partial sums; the distributed algorithm is unchanged,
  which keeps the GPU and CPU results directly comparable. (Copying the partial
  sums to the host before `MPI_Allreduce` is the portable path; on a
  CUDA-aware MPI build the device pointers could be passed to `MPI_Allreduce`
  directly, avoiding the staging copy — a possible optimisation.)

## 6. Validation (correctness)

**Parallel vs serial.** With identical seeds, the parallel versions reproduce the
serial baseline.

- MNIST, `K=10`, 8 MPI ranks: assignments **100% identical** to serial
  (`scripts/compare_labels.py` → raw and best-permutation agreement both
  `1.0000`), final inertia `1.784862e+11` in both, same 13 iterations (verified
  by re-running serial and MPI on the 70,000×784 MNIST set). The GPU version uses
  the same algorithm and reduction.
- Large blobs (`n=3,000,000, d=64, K=128`), 32 fixed iterations: the final
  inertia **agrees to six significant figures** across serial, MPI (1–16 ranks),
  and GPU — serial/MPI `8.624057e+08`, some parallel configurations
  `8.624058e+08` — a relative difference of about `1.2×10⁻⁷` (the values differ
  only in their seventh displayed digit). Assignment agreement with the serial
  baseline is very high but not bit-exact at this large `K`; on the small blobs
  set (`n=50,000, K=15`), where we re-ran the comparison directly, serial and MPI
  agree on `99.7–99.9%` of labels depending on rank count, with the mismatches
  confined to boundary points (see below).

**On the floating-point difference.** The large-`K` case does not reach a
bit-exact 100% label match, and this is expected, correct behaviour rather than a
bug. It is a direct consequence of a fundamental property of floating-point
arithmetic: because reals are stored with finite precision, floating-point
addition is **not associative**, i.e. `(a + b) + c ≠ a + (b + c)` in general.
A reduction such as `MPI_Allreduce` (and the GPU `atomicAdd`) therefore produces
slightly different least-significant bits depending on the order in which the
per-rank/per-thread partial sums happen to be combined — and that order varies
with the number of ranks and with scheduling. The lecture material flags exactly
this: `MPI_Reduce`/`MPI_Accumulate` "will likely produce different answers across
different runs, even with the same input."

In K-means the effect is tiny and bounded: the centroid coordinates differ only
in their last float bits, so only points lying almost exactly on the boundary
between two centroids can flip assignment. On the small blobs set we measured
this directly — serial vs 2-rank MPI flips about 0.3% of labels, serial vs
4/8-rank MPI about 0.08% — and in every case the aggregate inertia stays within a
relative `~10⁻⁶`, so the clusterings are numerically equivalent and only
genuinely ambiguous points differ. (We did not save the 3-million-point blob
label files, so the exact flip count at the largest `K` is not reported here; the
six-significant-figure inertia agreement is the quantitative evidence of
equivalence.) Full bit-for-bit reproducibility would require a deterministic
fixed-order reduction (for example, gathering all partial sums and summing them
in a canonical order, at some performance cost); we judged the six-significant-
figure agreement more than sufficient for a clustering task and did not impose
that overhead.

**Against ground truth.** The synthetic blobs carry planted labels, giving a
direct oracle. On the large dataset, purity is **0.977** with **126 of 128
clusters perfectly pure** (the two imperfect clusters are a k-means++
local-minimum artefact where two planted blobs merged and one split — identical
across all implementations). On MNIST digit labels, purity is ≈ 0.585 with
`K=10` on raw pixels (Section 9).

**Validation on the cluster.** The Atlas runs reproduce the local results: across
serial and MPI at 1–16 ranks on Atlas, the final inertia is `8.624057e+08` or
`8.624058e+08` (agreeing with the local serial/MPI runs and the GPU run to six
significant figures, relative difference `~10⁻⁷`). Correctness is therefore
preserved not only across parallelisation strategies but across platforms, all
using Open MPI 4.1.x (4.1 locally, 4.1.6 on Atlas and Vanda), compiled
independently on each machine.

## 7. Parallel scaling and efficiency (local reference)

This section characterises the scaling behaviour on a local workstation. These
numbers are the **preliminary reference**; the required cluster figures are
reported separately in Section 8 (Atlas / Vanda).

> **Test platform (local):** AMD Ryzen 7 6800HS — **8 physical cores, 16 logical
> threads** (2-way SMT) — with ~22 GB usable RAM, Open MPI 4.1, GCC 13, and a
> single NVIDIA RTX 3050 Ti Laptop GPU (compute capability 8.6, 4 GB VRAM, 20
> SMs). Each point is stored as a single-precision `float`. The distinction
> between 8 physical cores and 16 threads is important for reading the scaling
> below: ranks 1–8 each get a dedicated core, whereas 9–16 share cores via SMT.

We measure **strong scaling**: the problem size is held fixed while the number of
workers increases, so speedup shows how much faster a *fixed* job runs with more
cores. (The complementary notion, **weak scaling**, grows the problem with the
worker count — Gustafson's view — and is noted as an untested extension at the
end of this section.) The dataset is a large synthetic set
(`n=3,000,000`, `d=64`, `K=128`), run for a **fixed 32 iterations**
(`--fixed-iter`) so that every configuration does identical work and timing is
not confounded by convergence differences. Speedup `S(p) = T_serial / T(p)`;
efficiency `E(p) = S(p)/p`.

**Table 7.1 — MPI (CPU) strong scaling, local workstation.**

| MPI processes | Time (s) | Speedup | Efficiency | Core usage |
|---------------|----------|---------|------------|------------|
| 1 (serial)    | 597.9    | 1.00    | 1.00       | 1 physical core |
| 2             | 313.4    | 1.91    | 0.95       | dedicated cores |
| 4             | 163.9    | 3.65    | 0.91       | dedicated cores |
| 8             | 87.8     | 6.81    | 0.85       | all 8 physical cores |
| 16            | 65.4     | 9.14    | 0.57       | 2 ranks/core (SMT) |

The dataset is sized so the serial baseline runs ~10 minutes, keeping compute
well above measurement noise and MPI start-up overhead.

![Strong-scaling speedup (local)](../results/speedup.png){ width=60% }

![Strong-scaling efficiency (local)](../results/efficiency.png){ width=60% }

**Analysis.** Up to 8 ranks — the number of **physical** cores — efficiency stays
high (≈ 0.85–0.95): the assignment work divides cleanly across dedicated cores
and the `O(K·d)` reduction is small next to per-process compute. The gentle
decline from 2→8 (0.95→0.85) reflects the growing relative cost of the
per-iteration `MPI_Allreduce` (which moves `K·d = 128·64 = 8192` floats plus
counts every iteration) and shared memory-bandwidth pressure as all cores become
active. At 16 ranks efficiency drops sharply to ≈ 0.57, but this is **not** a
limitation of the algorithm: the Ryzen 7 6800HS has only **8 physical cores**, so
16 ranks place two ranks on each core via simultaneous multithreading (SMT).
SMT shares one core's arithmetic units between two threads, which gives little
extra throughput for this compute-bound inner loop — hence the fall-off. The
meaningful local scaling range is therefore 1→8; the 16-rank point is included to
show the expected SMT/over-subscription knee. Genuine scaling beyond 8 physical
cores requires the multi-node clusters (Atlas / Vanda), tested in Section 8,
where each rank can be placed on its own dedicated core.

*Weak scaling (untested extension).* We measured strong scaling only. A weak-scaling
study — growing `n` in proportion to the worker count so each worker keeps a
constant local load — would be the Gustafson-style complement. Because the
per-iteration communication is `O(K·d)` and independent of `n` (Section 2.2),
weak scaling would be expected to hold efficiency even better than strong
scaling, since the fixed reduction cost is amortised over a growing per-worker
compute load; we did not run it as the project calls for strong-scaling figures.

### 7.1 GPU performance (MPI + CUDA, local)

On the single local GPU with the same problem and 32 fixed iterations:

**Table 7.2 — Serial vs CPU-MPI vs single-GPU, local workstation.**

| Configuration               | Time (s) | Speedup vs serial |
|-----------------------------|----------|-------------------|
| Serial (1 core)             | 597.9    | 1.0×              |
| MPI, 8 ranks (8 cores)      | 87.8     | 6.8×              |
| MPI, 16 ranks (8 cores+SMT) | 65.4     | 9.1×              |
| MPI + CUDA, 1 GPU           | 35.2     | **17.0×**         |

A single GPU beats even the fully-loaded CPU because the assignment step
(`3e6 × 128 × 64` distance evaluations per iteration) is exactly the kind of
regular, data-parallel arithmetic a GPU is built for. The design points from
Section 5.3 — upload points once, stage centroids in shared memory, accumulate
with `atomicAdd` — keep per-iteration overhead to the `O(K·d)` transfers.

## 8. Runtime figures from Atlas / Vanda

The required cluster runtime figures come from **Atlas**, which has large
multi-core CPU nodes (80 cores per node) and uses the PBS scheduler. The MPI
(CPU) strong-scaling study below was run there, with ranks placed on dedicated
physical cores. Because of queue contention on this shared cluster, the run that
reached the highest core count used two nodes (16 cores on each); this means the
per-iteration `MPI_Allreduce` crosses the node interconnect at 32 ranks, which
makes the sustained efficiency all the more notable. (Scheduling constraints are
discussed in Section 11.)

Runs were submitted with `pbs/atlas_sweep.pbs`, which executes the whole 1→32
sweep in one job and writes `results/scaling_atlas.csv`; see README.

### 8.1 Cluster and environment

| Item | Atlas (CPU) |
|------|-------------|
| Cores per node | 80 physical cores |
| Scheduler | PBS Pro (`qsub`) |
| Compiler | GCC 13.2.0 |
| MPI implementation | Open MPI 4.1.6 (`OpenMPI/4.1.6-GCC-13.2.0`) |
| Dataset | 3,000,000 points × 64 dims, K = 128, 32 fixed iterations |
| Run placement | 2 nodes × 16 cores, one rank per dedicated core |

> The MPI + CUDA (GPU) study was run on **Vanda** (Vanda has GPU nodes; Atlas is
> CPU-only). The multi-GPU (1→2 A40) scaling figures are reported in Section 8.3;
> the local single-GPU result in Section 7.1 additionally demonstrates the CUDA
> implementation on consumer hardware.

### 8.2 MPI (CPU) strong scaling on Atlas

Measured on Atlas in a two-node allocation of 16 cores per node (32 slots total),
same dataset and 32 fixed iterations as the local study. Because the ranks fill
the first node before spilling onto the second, the 1–16-rank rows all run
*within a single node* and only the 32-rank row spans both nodes (see the
analysis below). Speedup `S(p) = T_serial / T(p)`; efficiency `E(p) = S(p)/p`,
baseline = Atlas serial (404.6 s).

**Table 8.1 — MPI (CPU) strong scaling, Atlas (2-node allocation, 16 cores/node; 1–16 ranks on one node, 32 ranks across two).**

| MPI processes | Time (s) | Speedup | Efficiency |
|---------------|----------|---------|------------|
| 1 (serial)    | 404.6    | 1.00    | 1.00       |
| 2             | 202.4    | 2.00    | 1.00       |
| 4             | 100.9    | 4.01    | 1.00       |
| 8             | 50.5     | 8.01    | 1.00       |
| 16            | 25.3     | 15.97   | 1.00       |
| 32            | 12.9     | 31.32   | 0.98       |

![Strong-scaling speedup, Atlas](../results/speedup_atlas_2node.png){ width=60% }

![Strong-scaling efficiency, Atlas](../results/efficiency_atlas_2node.png){ width=60% }

**Analysis.** On Atlas the implementation scales almost ideally: efficiency stays
at ≈ 1.00 up to 16 ranks and is still **0.98 at 32 ranks**, for a 31.3× speedup.
The measured speedup line is visually indistinguishable from the ideal-linear
line across the whole range. Inertia agrees to six significant figures at every
process count (`8.624057e+08`/`8.624058e+08`), confirming correctness is
preserved as the job is parallelised.

Two points make this result strong. First, up to 16 ranks the job runs within a
single node, so the `MPI_Allreduce` uses fast shared memory; the near-unity
efficiency there shows the per-rank work divides cleanly. Second — and more
telling — the 32-rank point spans **two nodes**, so the reduction now crosses the
network interconnect, yet efficiency only slips from 1.00 to 0.98. This is a
direct consequence of the design choice in Section 2.2: the reduction moves just
`K·d = 128·64 = 8192` floats per iteration regardless of `n`, so even inter-node
communication is negligible next to the `O(n·K·d)` assignment work. The algorithm
is firmly compute-bound, and the parallelisation adds almost no overhead even
across nodes.

The contrast with the local workstation (Section 7) completes the picture.
There, efficiency collapsed to ≈ 0.57 at 16 ranks because the laptop has only 8
physical cores, so 16 ranks contended two-to-a-core via SMT. On Atlas each rank
runs on its own physical core and efficiency stays near 1.0 to 32 ranks. The
laptop fall-off was therefore a property of the test hardware, not of the
algorithm: given dedicated cores, the method scales linearly. Higher counts
still (64, up to the per-job limit of 96 cores) are a natural extension but were
bounded by scheduler availability on this shared cluster (Section 11).

### 8.3 MPI + CUDA (GPU) scaling on Vanda

The MPI + CUDA implementation was built and run on **Vanda**, on a node with two
**NVIDIA A40** GPUs (Ampere, compute capability 8.6), using the same dataset and
32 fixed iterations as every other measurement. The build used
`GCC 13.2.0 + OpenMPI 4.1.6 + CUDA 12.4` (see `pbs/vanda_cuda.pbs`). One MPI rank
is bound to one GPU (`cudaSetDevice(rank % deviceCount)`), so the 1- and 2-GPU
runs use an identical code path and differ only in rank count.

**Table 8.2 — MPI + CUDA GPU strong scaling, Vanda (NVIDIA A40).**

| GPUs | Time (s) | Speedup vs 1 GPU | Efficiency | Speedup vs Atlas serial |
|------|----------|------------------|------------|-------------------------|
| 1    | 8.62     | 1.00             | 1.00       | 46.9×                   |
| 2    | 4.31     | 2.00             | **1.00**   | **93.9×**               |

![Vanda multi-GPU strong scaling: 2 A40s halve the runtime for a 2.00× speedup at 1.00 efficiency. Both GPUs are on one node, so the per-iteration `MPI_Allreduce` stays on-device/on-node.](../results/vanda_gpu_scaling.png){ width=92% }

Two GPUs run the benchmark in **4.31 s versus 8.62 s on one** — an exactly
**2.00× speedup, i.e. 1.00 parallel efficiency**. The final inertia is
`8.624058e+08`, agreeing with the 1-GPU result and every CPU run to six
significant figures (relative difference `~10⁻⁷`), so doubling the GPU count
preserves the clustering. Both
A40s sit on a single node, so the per-iteration `MPI_Allreduce` that combines the
two GPUs' partial sums stays within the node; the near-ideal efficiency confirms
that this cross-GPU reduction — again just `K·d = 8192` floats — is negligible
next to the halved `O(n·K·d)` assignment work. At ≈ 0.13 s per iteration, two
A40s are roughly **94× faster than a single Atlas CPU core** and about 3× faster
than all 32 Atlas CPU cores combined (12.9 s, Section 8.2).

This directly validates the `O(K·d)` scaling argument on the GPU path: the same
property that kept the Atlas CPU runs efficient across nodes keeps the GPU runs
efficient across devices. Scaling beyond two GPUs (4 GPUs across two nodes, where
the reduction would cross the node interconnect) is the natural next step and
would be expected to hold until the per-GPU work shrinks enough that even the
tiny inter-node reduction becomes visible; it was not run because a 4-GPU / 2-node
allocation did not free up within the project window (Section 11.2).

### 8.4 Cross-platform summary

Best measured runtime for each implementation on the same problem
(`n = 3,000,000`, `d = 64`, `K = 128`, 32 fixed iterations):

**Table 8.3 — Best runtime per implementation and platform.**

| Implementation | Platform | Best config | Time (s) | Speedup vs serial |
|----------------|----------|-------------|----------|-------------------|
| Serial         | Atlas    | 1 core      | 404.6    | 1.0×              |
| MPI (CPU)      | Laptop   | 16 ranks / 8 cores | 65.4 | 9.1×            |
| MPI (CPU)      | Atlas    | 32 ranks (2 nodes × 16) | 12.9 | 31.3×       |
| MPI + CUDA     | Laptop   | 1 GPU (RTX 3050 Ti) | 35.2 | 11.5×           |
| MPI + CUDA     | Vanda    | 1 GPU (A40)         | 8.62 | 46.9×           |
| MPI + CUDA     | Vanda    | 2 GPUs (A40)        | 4.31 | **93.9×**       |

![Best runtime for each implementation and platform (log scale; speedup vs the Atlas serial core annotated). Two A40s (4.31 s) are the fastest option, well ahead of all 32 Atlas CPU cores (12.9 s).](../results/cross_platform_runtime.png){ width=75% }

Three takeaways. First, with dedicated cores the CPU implementation scales
linearly (Atlas, 31.3× on 32 ranks at 0.98 efficiency — even across two nodes)
where the SMT-limited laptop could not (9.1× on 16 ranks over 8 cores). Second,
the GPU is the fastest option by a wide margin and scales cleanly: a single
datacenter A40 (46.9×) already beats all 32 Atlas CPU cores (31.3×), and two
A40s (93.9×, 2.00× over one GPU at 1.00 efficiency) are about 3× faster still —
even a modest laptop GPU (11.5×) is competitive with many CPU cores, underlining
why the distance-bound assignment step is the right target for GPU offload.
Third, every configuration converges to the same inertia
(`8.624057e+08`/`8.624058e+08`, agreeing to six significant figures), so the ~94×
range in runtime is pure performance, with correctness preserved throughout.

## 9. Analysis and discussion

The measured results (Sections 7–8) line up closely with what the design in
Sections 1–5 predicts. This section steps back from the individual numbers to
draw out *why* the behaviour looks the way it does, and what it says about the
parallelisation.

### 9.1 The algorithm is compute-bound, and that is why it scales

The single most important observation across every platform is that K-means, as
parallelised here, is **compute-bound, not communication-bound**. The design
decision in Section 2.2 — reduce the `O(K·d)` partial sums rather than the
`O(n)` assignments — fixes the per-iteration communication at `K·d = 8192`
floats regardless of how large the dataset is or how many workers share it. The
assignment work, by contrast, is `O(n·K·d)` and dominates.

The Atlas numbers make this concrete. Efficiency holds at ≈ 1.00 through 16 ranks
*within a node* and is still **0.98 at 32 ranks across two nodes**, where the
reduction now crosses the network interconnect. A communication-bound algorithm
would show a clear penalty the moment the reduction left shared memory; here the
inter-node step costs almost nothing because it is amortised over a large block
of local compute. This is Amdahl's law (Section 1.4) working in our favour: the
serial fraction `S` — the reduction plus its synchronisation — is tiny, so the
achievable speedup stays close to linear.

### 9.2 Hardware, not the algorithm, set the ceilings we hit

Two apparent "slow-downs" in the data are properties of the test hardware, not of
the method:

- **The laptop's 0.57 efficiency at 16 ranks** is simultaneous multithreading, not
  an algorithmic limit. The Ryzen 7 6800HS has 8 physical cores; ranks 9–16 share
  arithmetic units two-to-a-core, which gives little extra throughput for a
  compute-bound inner loop. Atlas, giving each rank a dedicated physical core,
  stays near 1.00 efficiency over the same range — the clean control that proves
  the point. The lesson is a general one from the lecture material: speedup is
  bounded by the number of *physical* execution units, and oversubscription past
  them buys little for arithmetic-heavy work.

- **The single-GPU result on Vanda was insensitive to host CPU count.** We ran the
  MPI+CUDA build with `ncpus=8` and again with a single host core, and the cluster
  time was unchanged at ≈ **8.62 s** (versus 8.63 s earlier). This is expected:
  with one MPI rank driving one GPU, the assignment arithmetic lives on the A40,
  so adding host cores cannot help — the host only launches kernels and runs the
  `O(K·d)` reduction. It is a useful negative result because it points at the
  *right* lever for more GPU performance: more GPUs (more ranks, each bound to a
  device), not more host cores. The 2-GPU run (Section 8.3) confirms this
  directly — doubling the *GPUs* halved the runtime (2.00×), the scaling that
  doubling the *host cores* could not deliver.

### 9.3 Why the GPU wins so decisively

A single datacenter A40 (8.62 s, 46.9× over an Atlas core) beat **all 32 Atlas
CPU cores combined** (12.9 s), and two A40s (4.31 s, 93.9×) widened the gap to
roughly 3×. This is the expected outcome for a distance-bound kernel. The assignment step is regular, branch-light floating-point arithmetic
over a large array — precisely the SIMD-style workload a GPU is built for
(Section 2.3). The Section 5.3 design choices are what let the GPU approach its
potential: uploading the points once keeps the slow PCIe link out of the inner
loop, staging centroids in shared memory turns `K` global reads per thread into
one, and bounding `atomicAdd` contention by `K` (not `n`) keeps the reduction
cheap. Without those, the kernel would be throttled by memory movement long
before it ran out of arithmetic throughput.

![Cost of a single Lloyd iteration: 12.6 s on one serial core collapses to 0.40 s across 32 Atlas cores, 0.27 s on one A40 GPU, and 0.13 s on two A40s.](../results/per_iter_time.png){ width=60% }

### 9.4 Correctness held everywhere, and the small differences are understood

Across three implementations, three platforms (laptop, Atlas, Vanda) all running
Open MPI 4.1.x, and process counts from 1 to 32, the final inertia agreed to
**six significant figures** (`8.624057e+08`/`8.624058e+08`, relative difference
`~10⁻⁷`), and the repeated Vanda run reproduced its value. The small label
differences on the large-`K` case are not a defect but the expected consequence
of floating-point non-associativity (Section 6): different reduction orders
combine the same partial sums in a different sequence, flipping only the handful
of points that sit almost exactly on a cluster boundary (measured at `0.1–0.3%`
on the small blobs set; not separately counted for the 3-million-point run, whose
label files were not retained). That the aggregate inertia is nonetheless stable
to six significant figures is the strongest statement of correctness available
for a reduction-based parallel algorithm.

### 9.5 What the results do and do not establish

The strong-scaling evidence is solid: given dedicated cores, the CPU
implementation scales near-linearly to 32 ranks even across nodes; the GPU path
is correct and fast; and the GPU path now also has a direct multi-device data
point — 2 A40s at 2.00× speedup and 1.00 efficiency (Section 8.3) — which
confirms the `O(K·d)` scaling argument holds across GPUs, not just across CPU
nodes. The one thing the experiments still do **not** establish — because of the
resource constraints in Section 11 — is the shape of the curve at higher counts:
single-node CPU beyond 32 cores (toward the 96-core per-job limit) and GPU beyond
2 devices (4 GPUs across two nodes, where the reduction would cross the
interconnect). The `O(K·d)` argument predicts both would continue to scale well,
and the measured 2-node Atlas (0.98 at 32 ranks) and 2-GPU Vanda (1.00) results
are strong direct support, but the higher counts themselves were not measured and
so remain an expectation rather than a result.

## 10. Conclusions about the data (MNIST)

The required "conclusions about real data" are drawn from MNIST, the real dataset
used in this project (70,000 handwritten-digit images, each a 784-dimensional
raw-pixel vector, with the true digit 0–9 available as a ground-truth label).

Clustering raw-pixel MNIST with `K=10` reaches ≈ 0.585 purity. The per-cluster
majority classes show *which* digits are separable in raw pixel space:

- Clean clusters: **0** (majority fraction ≈ 0.94), **2** (≈ 0.91), **6** (≈ 0.86).
- Confused: **4/7/9** and **3/5/8** share pixel-level structure (loops and strokes
  in similar positions) and split or merge across clusters.

Conclusion: Euclidean K-means on raw pixels captures gross shape but cannot
separate digits whose ink overlaps heavily. This matches the known result that
MNIST benefits from a feature transform (PCA, learned embeddings) before
clustering — a natural extension, but outside this project's parallelisation
focus.

## 11. Limitations

The implementation and its validation are complete, but the *scope of the
cluster experiments* was constrained by shared-resource availability rather than
by the code. These constraints are documented here for transparency; none of
them affects the correctness results in Section 6.

### 11.1 Atlas (CPU): scheduling bounded the core counts

Atlas is a heavily shared cluster, and the `parallel` queue imposes per-job
limits (maximum 96 cores per job, minimum 12 cores per chunk, maximum 8 running
jobs). The binding constraint in practice, however, was not these limits but
live availability: a request for a large block of cores on a single node could
not be satisfied for an extended period.

Concretely, requests for 80 or even 32 cores on one node (`select=1`) remained
queued for up to two days because no single node ever had that many cores free
at once (the scheduler reported, e.g., 80 requested vs ~50 available on the
best-placed node). Small, easy-to-place requests scheduled almost immediately.
We therefore have two Atlas datasets: a **single-node** sweep that scheduled
reliably up to **16 ranks** (`results/scaling_atlas.csv`: 16 ranks → 15.73×
speedup, 0.983 efficiency — near-ideal), and the **two-node, 32-slot** allocation
reported in Section 8.2, which extended the curve to 32 ranks (15.97× at 16,
31.32× at 32). Single-node counts above 16 were valid requests but not obtainable
within the project window. The two-node allocation is not a flaw but it does
change the 32-rank measurement, since the reduction phase then crosses the node
interconnect rather than shared memory; we therefore describe the 32-rank point
as a two-node result rather than a continuation of the clean single-node curve.

### 11.2 Vanda (GPU): multi-GPU scaling bounded at two devices

The MPI + CUDA implementation was built and run on Vanda across **one and two
NVIDIA A40 GPUs** (Section 8.3): both counts reproduced the CPU inertia to six
significant figures, and two GPUs scaled at 2.00× (1.00 efficiency) over one. The
toolchain (`GCC 13.2.0`, `OpenMPI 4.1.6`, `CUDA 12.4`) worked as expected. The
remaining bound is on **scaling beyond two GPUs**:

1. **Two GPUs were obtained on a single node**, which gives a clean intra-node
   measurement but means the 2-GPU reduction never leaves the node. A 4-GPU run
   spanning two nodes — which would exercise the inter-node interconnect on the
   GPU path, the GPU analogue of the 32-rank Atlas run — needs a larger GPU
   allocation.
2. **GPU nodes are heavily contended.** Securing an allocation with four GPUs, or
   two 2-GPU nodes at once, requires the scheduler to place a larger GPU request
   on a busy partition; this did not free up within the project window. (Vanda's
   login node is also a GPU-less VM, so all building and running must happen on a
   scheduled compute node.)

The 1- and 2-GPU results are therefore complete, validated data points that
establish near-ideal GPU scaling; only the 4-GPU / 2-node extension is
outstanding. The code is ready for it via `pbs/vanda_cuda.pbs` — with one rank
bound to one GPU (`qsub -l select=2:ncpus=2:ngpus=2:mpiprocs=2 -v NP=4`), it
reuses the same `MPI_Allreduce` used everywhere else.

### 11.3 Summary

The project's required deliverables — serial, MPI, and MPI+CUDA implementations;
correctness validation; CPU scaling on a cluster (Atlas, 1→32 ranks); and
multi-GPU CUDA scaling on Vanda (1→2 A40s) — are all satisfied. The two items
bounded by resource availability are higher single-node CPU core counts (beyond
32, toward the 96-core limit) and GPU scaling beyond two devices (4 GPUs across
two nodes); both are immediately runnable with the included PBS scripts should
cluster time become available.

## 12. Conclusion

We implemented Lloyd's K-means three ways — a serial baseline, a distributed-
memory MPI version, and an MPI+CUDA version — around a single data-parallel
decomposition over points, with the centroid update expressed as a global
reduction phase (three `MPI_Allreduce` collectives of size `O(K·d)` per
iteration). The reduction volume is independent of `n`, which is what lets the
method stay compute-bound and scale well.

The measurements bear this out on a 3-million-point, 64-dimensional, `K=128`
benchmark at a fixed 32 iterations. On Atlas the MPI version scales near-linearly
to 32 ranks (31.3×, 0.98 efficiency, even across two nodes); on Vanda the
MPI+CUDA version runs a single A40 in 8.62 s and two A40s in 4.31 s (2.00×, 1.00
efficiency), the fastest configuration measured and about 94× an Atlas serial
core. Correctness is preserved throughout: serial and MPI are bit-identical at
one rank, MNIST clustering is 100% label-identical between serial and 8-rank MPI,
and the benchmark inertia agrees to six significant figures across every
implementation, platform and worker count, with the only differences being the
expected last-bit effects of non-associative floating-point reduction. On real
MNIST data, raw-pixel K-means reaches 0.585 purity, cleanly separating visually
distinct digits (0, 2, 6) while merging stroke-similar ones (4/7/9, 3/5/8).

The work left undone is bounded by shared-cluster availability rather than by the
code: single-node CPU runs beyond 32 cores and GPU scaling beyond two devices
(Section 11). Both are immediately runnable with the included scripts.

## 13. AI usage declaration

An AI coding assistant (Kiro, built on a large language model) was used as a
support tool throughout this project: for scaffolding the repository structure;
drafting and debugging the serial, MPI and MPI+CUDA C++ sources and the
supporting Python/Bash scripts; drafting and editing portions of this report and
the README; and preparing the diagrams, illustrations and figure-generation
code. It was also used to diagnose build and environment issues on the clusters
(for example the `nvcc` MPI-flag problem and the PBS `ebenv` setup) and to run
the local validation and scaling experiments.

The project team remains responsible for the design decisions, for running the
cluster experiments on Atlas and Vanda, for interpreting the results, and for
ensuring the submitted claims match the measured outputs. Where numbers in this
report could be re-checked from the code and data they were (serial-vs-MPI
agreement, MNIST purity and the per-cluster majority digits, the local and
cluster scaling tables); results that depend on artefacts we did not retain (the
exact label-flip count on the three-million-point blob set) are reported as such
rather than overstated. The team takes responsibility for the correctness and
integrity of the final submission, including clearly identifying the limitations
and the experiments that were not completed (Section 11).

## 14. References

1. S. Lloyd, "Least squares quantization in PCM," *IEEE Transactions on
   Information Theory*, vol. 28, no. 2, pp. 129–137, 1982.
2. D. Arthur and S. Vassilvitskii, "k-means++: The advantages of careful
   seeding," in *Proc. 18th ACM-SIAM Symposium on Discrete Algorithms (SODA)*,
   pp. 1027–1035, 2007.
3. Y. LeCun, C. Cortes and C. J. C. Burges, "The MNIST database of handwritten
   digits," http://yann.lecun.com/exdb/mnist/ (accessed October 2026).
4. Message Passing Interface Forum, *MPI: A Message-Passing Interface Standard,
   Version 4.0*, 2021. https://www.mpi-forum.org/docs/
5. NVIDIA Corporation, *CUDA C++ Programming Guide*, v12.4, 2024.
   https://docs.nvidia.com/cuda/cuda-c-programming-guide/
6. Project source code and scripts:
   https://github.com/AmritSatpathy/parallelKmean

## Appendix: reproducing results

See `README.md`. In short: `make all`, `bash scripts/fetch_mnist.sh` (or
`./bin/gen_blobs …` for the synthetic set), then the run / validate / scaling
commands. Every `RESULT` line and label file referenced above is produced by
those commands.
