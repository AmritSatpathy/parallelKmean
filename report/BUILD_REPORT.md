# Publication report

`Parallel_Kmeans_Project_Report.docx` is the editable designed report, and
`Parallel_Kmeans_Project_Report.pdf` is its verified submission PDF. The design
follows `assets/PROJECT_REPORT_DESIGN_TEMPLATE_GUIDE.md`, with a relaxed title
page and a navigation-only contents page. Contents links have no underlining
or dotted leaders. The provided `../project_assignment.md` (one level above
the repository) specifies required report content but no page limit; the build
does not target a fixed page count. `report.md` is generated
from the same document content so the Markdown, Word, and PDF editions agree.

The builder reads the CSV files without changing them. All runs are presented
as one team project, grouped by platform, workload, and measurement batch.
`experiment_manifest.json` records source artifact hashes and known scheduler
identifiers. The report and manifest list only completed experiments used in the
project. Local CPU and CUDA measurements are labelled “Local run”; HPC results
retain their platform labels and documented scheduler identifiers.

## Build and refresh navigation

Use a Python environment containing python-docx, matplotlib and PyMuPDF:

```bash
python3 report/scripts/build_kmeans_report.py
```

Render the DOCX with LibreOffice or the documents skill's `render_docx.py` to
produce a PDF. Then derive the contents page from actual heading locations:

```bash
python3 report/scripts/update_report_contents.py /path/to/rendered-report.pdf
```

Render again, confirm pagination is stable, and copy the verified PDF to
`report/Parallel_Kmeans_Project_Report.pdf`. The contents uses native Word
TOC/PAGEREF fields and heading bookmarks; the update utility caches page numbers
from the actual rendered document rather than manually typing them.

Figures are generated reproducibly as PNG and SVG in `figures_publication/`.
No K-means computation or HPC submission is performed during report generation.

## Evidence and interpretation

- All local/Atlas/Vanda measurements remain in their existing CSV files.
- Atlas jobs 1034257.venus01 (16 ranks), 1036751.venus01 (32 ranks), and
  1036776.venus01 (48 ranks), and Vanda jobs 1533324.stdct-mgmt-02 and
  1533344.stdct-mgmt-02 have downloaded evidence and separate result files.
- Added completed Atlas jobs 1036952.venus01 (12 ranks), 1037018.venus01
  (16 ranks), and 1037036.venus01 (24 ranks). Their CSVs and logs are copied
  into job-specific result and Atlas log paths without altering earlier inputs.
- The 100-iteration chart is one team series: points retain individual jobs,
  and lines use per-rank medians where the 16-rank configuration is repeated.
  Both 16-rank timing pairs remain visible in Table 6; no authorship grouping
  or replacement of an earlier completed measurement is used.
- The Atlas MNIST log for 1027948.venus01 is retained in pbs/logs/atlas; the Vanda MNIST test remains documented from the session record.
- Every 100-iteration CPU speedup uses its own job's serial baseline; all six
  downloaded Atlas logs confirm exit status 0 and matching CSV measurements.
- Failed and incomplete attempts in the download archive are excluded from
  the report, chart data, and experiment manifest. Nothing submits or reruns them.
- The 100-iteration Atlas and 32-iteration Vanda workloads use different iteration budgets and report different final inertia;
  cross-platform timing ratios are not presented as controlled speedups. The Atlas and Vanda datasets downloaded on 12 November have identical SHA-256 checksums; the large CSV is excluded from the code-only submission, with its checksum and generator instructions retained. Public MNIST downloads were also verified byte-identical; README documents downloading them. Job-time checksums were not recorded. Downloaded source and Makefiles match the repository; archived HPC snapshots and the reconciliation manifest are included.
- NUS policy reference: current version 15 September 2026,
  https://ctlt.nus.edu.sg/wp-content/uploads/2026/04/Policy-for-Use-of-AI-in-Teaching-and-Learning.pdf
- Citation guidance: https://libguides.nus.edu.sg/citation/ai
- Official full-colour logo source:
  https://nus.edu.sg/images/default-source/identity-images/NUS_logo_full-vertical.jpg
  (reused from the existing local report-template assets).

The team should review the final author declaration and all empirical claims
before submission. Tool names in the AI declaration are recorded as provided
by the authors.
