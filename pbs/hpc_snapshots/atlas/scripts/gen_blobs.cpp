// gen_blobs.cpp
// Fast C++ generator for synthetic Gaussian-blob clusters with known ground
// truth. Equivalent in spirit to scripts/gen_blobs.py but orders of magnitude
// faster, so it can produce multi-million-point datasets in seconds.
//
// Output CSV: first column is the true cluster id (0..k-1), remaining columns
// are the feature coordinates. Use --label-col 0 with the K-means programs.
//
// Usage:
//   ./gen_blobs --n 2000000 --dims 64 --k 128 --out data/blobs_big.csv
//               [--spread 1.0] [--sep 20.0] [--seed 0]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    long long n = 2'000'000;
    int dims = 64, k = 128;
    double spread = 1.0, sep = 20.0;
    unsigned seed = 0;
    std::string out = "data/blobs_big.csv";

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto nx = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (s == "--n") n = std::atoll(nx());
        else if (s == "--dims") dims = std::atoi(nx());
        else if (s == "--k") k = std::atoi(nx());
        else if (s == "--spread") spread = std::atof(nx());
        else if (s == "--sep") sep = std::atof(nx());
        else if (s == "--seed") seed = static_cast<unsigned>(std::strtoul(nx(), nullptr, 10));
        else if (s == "--out") out = nx();
        else { std::fprintf(stderr, "unknown arg %s\n", s.c_str()); return 2; }
    }

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> center_dist(-sep, sep);
    std::normal_distribution<double> noise(0.0, spread);
    std::uniform_int_distribution<int> pick_cluster(0, k - 1);

    // Random, well-separated cluster centers.
    std::vector<double> centers(static_cast<size_t>(k) * dims);
    for (auto& c : centers) c = center_dist(rng);

    FILE* f = std::fopen(out.c_str(), "w");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", out.c_str()); return 1; }

    // Large user-space buffer to reduce write syscalls.
    std::vector<char> iobuf(1 << 22);
    std::setvbuf(f, iobuf.data(), _IOFBF, iobuf.size());

    // Per-row text buffer. 64 dims * ~12 chars + label is well under this.
    std::string line;
    line.reserve(static_cast<size_t>(dims) * 16 + 16);
    char num[32];

    for (long long i = 0; i < n; ++i) {
        int cid = pick_cluster(rng);
        const double* c = &centers[static_cast<size_t>(cid) * dims];

        line.clear();
        int len = std::snprintf(num, sizeof(num), "%d", cid);
        line.append(num, len);
        for (int d = 0; d < dims; ++d) {
            double v = c[d] + noise(rng);
            len = std::snprintf(num, sizeof(num), ",%.5f", v);
            line.append(num, len);
        }
        line.push_back('\n');
        std::fwrite(line.data(), 1, line.size(), f);
    }

    std::fclose(f);
    std::fprintf(stderr, "wrote %lld points, dims=%d, k=%d -> %s\n",
                 n, dims, k, out.c_str());
    return 0;
}
