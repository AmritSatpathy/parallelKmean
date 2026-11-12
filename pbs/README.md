# HPC scripts in the submission

The scripts were compared against the actual Atlas and Vanda files downloaded by the team on 12 November 2026. All C++/CUDA sources, the Makefile, the generator and validation tools match the local repository. Original remote copies are retained under hpc_snapshots/atlas and hpc_snapshots/vanda; hpc_reconciliation.json records their hashes. The runnable scripts include the documented submission-time usability fixes below, so they are intentionally not byte-identical to the archived PBS scripts.

| Setting | Packaged configuration and evidence |
| --- | --- |
| Atlas compiler / MPI | GCC 7.3.0 and Open MPI 4.1.1, explicit PATH and library paths from the successful configuration; no broken `/app1/ebenv` or unavailable module load |
| Atlas build | Clean compute-node build; explicit g++ for serial/generator and mpicxx for MPI; source and compiler flags retained |
| Atlas default allocation | 16 CPUs / 16 MPI ranks / 40 GB, 2 hours; override allocation and PROCS together for other rank counts |
| Atlas output | Serial baseline, then specified MPI counts; fixed iteration budget; results/scaling_mpi.csv and scaling_atlas.csv |
| Vanda toolchain | GCC 13.2.0, Open MPI 4.1.6-GCC-13.2.0, CUDA 12.4.0; nvcc sm_86 matches the A40 logs |
| Vanda default allocation | One node, 36 CPUs, one rank, one GPU, 240 GB; two-GPU command in README requests 72 CPUs, two ranks and 480 GB |
| Vanda linking | Open MPI include/library directories supplied to nvcc by Makefile, as in retained build logs |
| PBS syntax | No inline trailing comments in directives; memory included in select |

Submission preparation also adds the missing data-directory creation on Vanda, makes the CPU harness return failure if a benchmark fails or lacks a RESULT line, and prevents a missing real-data CSV from silently becoming synthetic data. Automatic generation is allowed only for the default `data/blobs_big.csv`, with the generator's explicit defaults (seed 0, spread 1, separation 25). Existing nonempty inputs are reused. Clustering seed defaults to 42.

Successful Vanda logs contain ignored `libpbs.so.0` warnings for Open MPI TM components. We have not invented an unrecorded fix or suppressed these messages. The recipes target the demonstrated single-node one/two-GPU setup; they do not claim multi-node GPU validation. Site module availability, account permissions and scheduler limits can change.

Historical timings remain unchanged. These submission-time usability fixes were checked statically and with a stub failure test, not by rerunning HPC experiments. Use a distinct working directory per job to preserve inputs, binaries and output files.

The submission download script additionally verifies the published MNIST CSV checksums and installs downloads atomically. Dataset CSVs are excluded from the ZIP; README gives the download and generation sequence. Historical HPC snapshots retain the original scripts for comparison.
