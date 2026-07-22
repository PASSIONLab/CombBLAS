# Session Log

---

### 2026-07-07 — Swiss-Prot pipeline: similarity, HipMCL sweep, F-score setup

**Decisions**:
- **Similarity search**: LAST + PAM30. Ran with two threshold pairs — id50/cov80 and id30/cov70 — to compare later.
- **HipMCL sweep**: 4 inflations × 2 similarity graphs = 8 runs, per the HipMCL paper (I = 1.4, 2, 4, 6). Uses stock `mcl` binary from `Applications/MCL.cpp` — reads labeled triples directly, no MTX preprocessing.
- **Dropped `-base 1`** from `mcl` invocation: only meaningful for MTX input; ignored on triples path.
- **Ground truth for F-score**: Pfam-multiset (Antonio's suggestion from 2026-06-08). A protein's family identity is the sorted set of all its Pfam accessions; two proteins share a family iff their sets match. No-Pfam proteins (24,692 of 575,503) each become singleton families.
- **Implementation** (`Applications/Incremental/scripts/build_pfam_ground_truth.py`):
  1. Build `sequences/swissprot/swissprot_id_map.tsv` — full FASTA-header key (`sp|ACC|NAME`) → 1-based integer, sequential in FASTA order.
  2. Stream `uniprot_sprot.dat`, collect each protein's Pfam-set, group by sorted tuple, write one line per group with integer IDs.

**Status — artifacts in place**:
- `sequences/swissprot/swissprot_id_map.tsv`
- `evaluation/ground_truth/swissprot_pfam_families.clusters` (49,732 lines: 25,040 multi-protein families + 24,692 singletons)
- `experiments/hipmcl/*.clusters` — 8 HipMCL outputs (string labels)
- `experiments/hipmcl/*.clusters.int` — same 8 remapped to integer IDs via awk one-liner
- Scripts: `lastal-similarity.sh`, `hipmcl.sh`, `build_pfam_ground_truth.py`

---

### 2026-06-23 — Swiss-Prot as gold standard; replaces archaea+UniParc for quality

**What we decided**:
- **Swiss-Prot replaces the archaea + UniParc plan for the quality story.** Manually curated UniProtKB entries already carry Pfam/InterPro/GO cross-references, so no InterProScan run is needed — this directly resolves the blocker from the 2026-06-11 session.
- Swiss-Prot covers **paper claims 2 (quality) and 3 (faithfulness)** only. **Scalability (claim 1)** is decoupled and will use a different (larger) dataset — choice deferred to a separate conversation.
- Annotation source going forward is the Swiss-Prot flat file (`.dat`) — canonical UniProt format, parses cleanly with Biopython (`SeqIO.parse(f, "swiss")`), DR lines give Pfam/InterPro/GO IDs per entry. Smaller than the XML variant.

**What we accomplished**:
- Downloaded Swiss-Prot **Release 2026_02 (10-Jun-2026)** to `$CFS/m4293/taufique/imcl/sequences/swissprot/`:
  - `uniprot_sprot.fasta` (275 MB) — for `lastdb`/`lastal`
  - `uniprot_sprot.dat` (4.0 GB) — annotations for ground truth
  - `reldate.txt` — release notes
  - Both `.gz` originals kept
- Verified counts match across both files: **575,503 entries**.

---

### 2026-06-11 — Re-sourcing archaea annotations from IMG; deferring InterProScan

**What we learned**:
- May 16 InterProScan test on `test_small.fasta` had **no recorded runtime** — no log was captured.
- `test_small.fasta` = first 10 records of `isolates_proteins_archaea.fasta` (all from single archaeal genome `2868197823`), identical to first 10 of `splits/isolates_proteins_archaea_chunk_0001.fasta`.
- Sequence counts: archaea reference = **5,925,937**; each UniParc part p1–p10 = ~4.7–5.0M (avg ~4.83M); full UniParc active ≈ ~900M–1B across 200 parts.
- **IMG already runs Pfam (+TIGRFAM, COG, KEGG) on every archaeal/bacterial isolate genome** via the MGAP / IMG Annotation Pipeline (V5.x). Method = HMMER search at Pfam gathering threshold. Method unchanged in principle since the V4 SOP paper ([PMC4623924](https://pmc.ncbi.nlm.nih.gov/articles/PMC4623924/)); V5.1 uses Pfam v34 ([V5.x change log](https://sites.google.com/lbl.gov/imghelp/img-annotation-pipeline-v-5-x/change-log)).
- **Implication**: re-running InterProScan from scratch on 5.9M archaea sequences is likely unnecessary — per-protein Pfam annotations should already exist alongside IMG-distributed genomes.
- IMG distributes annotations as **per-genome tarballs** (FASTA + GFF + tab-delimited Pfam/TIGRFAM tables, keyed by GOLD Analysis Project ID), bulk-stageable via Globus ([Globus bulk download instructions](http://genomeportal.jgi.doe.gov/portal/help/download_bulk_withGlobus.jsf)).
- Entry point: JGI Data Portal ([data.jgi.doe.gov](https://data.jgi.doe.gov/)) → **IUIG (All IMG Unrestricted Isolate Genomes)** portal ([genome.jgi.doe.gov/portal/IUIG](https://genome.jgi.doe.gov/portal/IUIG/IUIG.home.html)), 143,796 genomes (Nov 2023 release). Archaea filterable via Domain attribute or release metadata TSV.
- Our existing archaea FASTA came from the Pavlopoulos-lab mirror (`http://62.217.122.124/img_isolate_genomes/`), which doesn't ship annotations — we need to re-source from IMG/Globus.
- Note: GOLD ([gold.jgi.doe.gov](https://gold.jgi.doe.gov/)) is a metadata catalog only, not a file repository — wrong starting point for downloads.

**Where we stopped**:
- On the IUIG portal exploring the download UI — haven't yet pinned down the exact click path for archaea-only filtering and bulk Globus staging.
- Drafted a short, broad email to Antonio asking how he'd cut down annotation cost for tens of millions of sequences; experimental setup kept as black-box context, methodology/fairness review deferred.

**Next step when picking up**:
- Finish the IUIG / JGI Data Portal walkthrough to nail the bulk-download recipe for archaea isolates with Pfam annotations.
- For the UniParc side: check whether UniProt/InterPro publishes precomputed InterPro matches that we can download directly instead of computing.

---

### 2026-06-08 — InterProScan output interpretation, biologist clarification

**What we learned**:
- Heard back from Antonio on 2026-06-03 on how to interpret InterProScan TSV output for ground-truth family labels
- **Column 5 is the family label**, not column 12. Column 5 is the native family/signature ID from whichever database produced the row (e.g. `PF01497`, `PTHR30535`, `SSF53807`). Column 12 is an optional cross-reference to InterPro and is often `-`
- **Each database (column 4) should be treated as its own ground truth.** InterProScan runs many member databases (Pfam, PANTHER, SUPERFAMILY, Gene3D, CDD, Hamap, PIRSF, NCBIfam, …) and their column-5 IDs live in different naming systems and are not directly comparable
- **Some databases annotate domains, not whole proteins** — notably Pfam and CDD. A multi-domain protein produces one row per domain from those databases, so the protein's "label" under such a database is a set of domain IDs, not a single ID. Antonio's suggested evaluation: check whether proteins within a cluster share the same set of domains
- Grounded each point on the existing `test_small.tsv` (10 proteins, 54 rows). Concrete multi-domain example: protein `2868197823` has two Pfam rows — `PF01497` at residues 5–159 and `PF18204` at residues 230–250

---

### 2026-06-04 — CombBLAS label vector I/O gap, prototype fix

**What we accomplished**:
- Audited the CombBLAS vector I/O surface for label support (`FullyDistVec<IT, std::array<char, MAXVERTNAME>>`)
- Identified two incompatible file formats: `ReadDistribute`/`SaveGathered` (MatrixMarket-style `numrows numcols nnz` header, `row col val` lines, single-master, scalability bottleneck) vs `ParallelRead`/`ParallelWrite` (simpler `glen gnnz` header, `idx val` lines, fully parallel via MPI-IO)
- Identified the asymmetry: `ParallelWrite` accepts a `HANDLER`, `ParallelRead` does not — its `ss >> ii >> vv` won't compile for `std::array<char, N>` (no `operator>>`)
- Noted that no central label handler exists in CombBLAS; `StdArrayReadSaveHandler` lives only in `ReleaseTests/ParIOTest.cpp` (similar local handlers in `WriteMCLClusters.h`, `VectorIO.cpp`)
- Confirmed iMCL's existing workaround in `Main.cpp`/`Full.cpp`/etc.: convert labels to integers via `convLblToNum` before write, `convNumToLbl` after read — sidesteps the type issue but loses real label identifiers
- Flagged latent bug in `ReleaseTests/VectorIO.cpp`: uses `FullyDistVec<int64_t, std::string>` with `ParallelRead`. Compiles because `std::string` has `operator>>`, but `SparseCommon` ships raw bytes via `MPI_Alltoallv` — sends pointer values not string contents. Not in CTest, never run multi-rank
- Implemented minimal fix in `include/CombBLAS/FullyDistSpVec.cpp`: added `if constexpr (std::is_same_v<NT, std::array<char, MAXVERTNAME>>)` branch inside both parse loops in `ParallelRead`. Numeric path unchanged. Required `<type_traits>` and `<array>` (both already transitively included)
- Discovered second latent bug in `StdArrayReadSaveHandler::read`: declared `LBL strarray;` instead of `LBL strarray{};`, so tail bytes [n+1..63] are uninitialized garbage. Same issue in the new `if constexpr` branch (`vv` declared once outside the loop). `std::array::operator==` compares all 64 bytes so this caused first round-trip test to FAIL with v1 != v2 despite matching labels
- Fixed by value-initializing `strarray` in the local handler copy and adding `vv.fill('\0')` before each parse in the `if constexpr` branch
- Wrote `Applications/Incremental/LabelVecIOTest.cpp`: round-trips a label vector via ReadDistribute → ParallelWrite → ParallelRead → element-wise compare. Uses TESTDATA/20outta100.txt as input (numeric values read as label strings). Added target `label-vec-io-test` to `Applications/Incremental/CMakeLists.txt` and SLURM driver `scripts/label-vec-io-test.sh`

---

### 2026-05-16 — InterProScan test run, iMCL binary design

**What we accomplished**:
- Confirmed InterProScan data bundle extraction completed successfully (17 database subdirectories in place)
- Created temp directory at `$CFS/m4293/taufique/imcl/interproscan-5.77-108.0/temp/`
- Ran successful InterProScan test run on `test_small.fasta` (10 sequences)
  - Output: `$CFS/m4293/taufique/imcl/evaluation/interproscan/img_reference/test_small.tsv` (54 rows)
  - Used `--iprlookup -t p` flags — essential for InterPro family annotations
- Discussed how to interpret InterProScan TSV output (15 columns, column 12 = InterPro accession)
- Sent test files to collaborators to ask how to interpret output as ground truth family labels
  - `$CFS/m4293/taufique/imcl/sequences/img_reference/test_small.fasta`
  - `$CFS/m4293/taufique/imcl/evaluation/interproscan/img_reference/test_small.tsv`
- Designed `iMCL` binary interface (two modes: init and update) and full data flow
- Created `iMCL.cpp` skeleton with argument parsing — compiles successfully
- Added `imcl` target to `CMakeLists.txt`
- Investigated CombBLAS for string label read/write — identified `StdArrayReadSaveHandler` in `ReleaseTests/ParIOTest.cpp`
- Confirmed `IncrementalMCLSquare` in `ParFriends.h` is already used in `IncrementalMCL`, all dependencies publicly accessible
- Discussed moving `IncrementalMCLSquare` out of CombBLAS into `IncClust.h` — feasible, only requires adding includes

**Where we stopped**:
- `iMCL.cpp` skeleton compiles but `runInit` and `runUpdate` are not yet implemented
- SLURM job array for InterProScan not yet written — waiting on collaborator response
- Test program for string label I/O and `IncrementalMCLSquare` not yet written

**Next step when picking up**:
- Write a small test program to verify string label read/write (`StdArrayReadSaveHandler`) and `IncrementalMCLSquare` on small inputs
- Implement `runInit` in `iMCL.cpp`
- Once collaborators respond, decide on ground truth label interpretation and write SLURM job array for InterProScan

---

### 2026-05-16 — InterProScan setup, archaea sequence splitting (earlier session)

**What we accomplished**:
- Confirmed InterProScan can be run via Podman on Perlmutter
- Pulled InterProScan image: `docker.io/interpro/interproscan:5.77-108.0`
- Downloaded InterProScan data bundle (6.4 GB) to `$CFS/m4293/taufique/imcl/interproscan-5.77-108.0/data/`
- Verified md5 checksum — OK
- Started extracting the data bundle (extraction in progress at end of session)
- Split archaea FASTA into 5926 chunks of 1000 sequences each → `$CFS/m4293/taufique/imcl/sequences/img_reference/splits/`
- Created `scripts/` directory at `$CFS/m4293/taufique/imcl/scripts/`
- Decided output structure for InterProScan runs: per-chunk outputs → `evaluation/interproscan/img_reference/chunks/`, final merged results → `evaluation/interproscan/img_reference/`

---

### 2026-05-14 — Stock taking, directory setup, sequence downloads

**What we accomplished**:
- Took stock of all loose ends and documented them
- Defined paper claims and evaluation strategy
- Created directory structure under `$CFS/m4293/taufique/imcl/`
- Downloaded archaea sequences from IMG and converted to FASTA
- Downloaded UniParc sequences (parts 1-10) and unzipped them
- Started investigating InterProScan installation on Perlmutter

---

### 2026-05-04 — Brainstorming user-facing pipeline interface

**Goal**: Design a simple, biologist-friendly interface that connects `Triples2MtxNLbls` → `Main.cpp` (iMCL) across incremental steps.

**Key insights**:
- Pipeline flow: user provides triples → `Triples2MtxNLbls` → 4-block MTX + labels → `Main.cpp` → clusters + summary MTX + labels → feeds back as M11/L11 next step
- Decided on state directory approach with `summary.mtx`, `labels.txt`, `clusters.out`
- MPI invocation: user runs with `mpirun` as with any MPI program — no special handling needed
- `Triples2MtxNLbls` needs L11 (not TriplesM11) for update runs — L11 from previous run is sufficient to reconstruct M11_labelMap
- Mixed triples file contains sequences from old, new, and neither (M3) — M3 sequences are folded into the new block

---

### 2026-03-10 — Re-establishing experimental baseline

**What we accomplished**:
- Reviewed codebase and scripts to recall what experiments were previously done
- Downloaded archaea and eukarya FASTA files from IMG
- Documented IMG format (tab-separated, not standard FASTA) and conversion command:
  ```bash
  awk -F "\t" '{print ">"$1"\n"$2}' isolates_proteins_archaea > isolates_proteins_archaea.fasta
  ```
- LAST similarity search setup documented (lastdb + lastal, PAM30, 50% identity, 80% coverage)
