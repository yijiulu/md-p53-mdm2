# p53–DNA & MDM2–RPL11 Molecular Dynamics Pipeline

[![CI](https://github.com/yijiulu/md-p53-mdm2/actions/workflows/ci.yml/badge.svg)](https://github.com/yijiulu/md-p53-mdm2/actions/workflows/ci.yml)
[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-blue.svg)](LICENSE)

Reproducible, GPU-portable all-atom molecular dynamics (MD), trajectory QC, MM/GBSA per-residue decomposition, and synchronised PyMOL trajectory rendering pipeline built on **OpenMM 8**, **AmberTools (`tleap` + `MMPBSA.py`)**, and **ParmEd**.

The pipeline models structural and energetic perturbations caused by cancer-associated hotspot missense variants in two zinc-coordinating macromolecular complexes:

1. **MDM2–RPL11 (`4XXB`)**: Human MDM2 zinc-finger domain (`293–334`, CCCC $\mathrm{Zn^{2+}}$ site) bound to ribosomal protein L11 (`13–178`), targeting hotspot `MDM2 R326` (UniProt `Q00987`; corresponds to `p.R332` in COSMIC Ensembl isoform `ENST00000258149`).
2. **p53 DBD dimer–DNA (`2AC0_dimer`)**: Symmetric human p53 DNA-binding domain homodimer (`94–291`, 2× CCCH $\mathrm{Zn^{2+}}$ sites) bound to a 12-bp response-element duplex (`5'-CGGGCATGCCCG-3'`), targeting DNA-contacting hotspot `TP53 R280` (supporting both homozygous and heterozygous dimer variants).

## Biophysical Specifications

Both complexes share an identical solvent and force-field stack to avoid force-field asymmetry across comparisons:

| System | PDB & Chains | Residues & Capping | Bonded Zinc Model (`ZAFF`) | Force Field, Solvent & HMR |
| :--- | :--- | :--- | :--- | :--- |
| **`4XXB`** | `4XXB` Chains `A` (RPL11) + `B` (MDM2) | RPL11 `13–178` (internal loop `139–153` rebuilt); MDM2 `293–334`; `ACE`/`NME` caps (`OXT` stripped); buffer ligands removed | Center ID 1 (`ZN1` + 4× `CY1`: MDM2 `C305, C308, C319, C322`), placed contiguously after Chain B | `ff14SB` + 4-point `OPC` (`12.0 Å` truncated octahedron) + `0.15 M KCl` (Li–Merz 12-6 OPC ions) + ParmEd `4 fs` HMR (`3.024 Da` solute H) |
| **`2AC0_dimer`** | `2AC0` Chains `A, B` (p53) + `E, F` (DNA) | Symmetric p53 `94–291` (`ACE`/`NME` caps); 12-bp DNA duplex (`5'` phosphate stripped) | 2× Center ID 2 (`ZN2` + 3× `CY2` `C176, C238, C242` + 1× `HE1` `H179 [ND1]`), each contiguous with its host chain | `ff14SB` + `DNA.OL15` + 4-point `OPC` (`12.0 Å` truncated octahedron) + `0.15 M KCl` + ParmEd `4 fs` HMR |

Key engineering choices:

- **Contiguous zinc-chain ordering**: Placing each bonded `ZN1`/`ZN2` residue immediately after its coordinating protein chain prevents ParmEd `HMassRepartition` from reordering atoms relative to `solv.prmtop`, verified automatically via atom-signature assertion.
- **Roe & Brooks (2020) 10-step relaxation**: Double-precision CPU energy minimisation (Steps 1, 3, 4, 5) guards against mixed-precision GPU overflow, followed by progressive harmonic Cartesian restraint release and a `1.0 ns` unrestrained NPT density-stabilisation stage at `2 fs` before switching to `4 fs` HMR production.
- **Cross-GPU portable checkpointing**: Production saves XML states (`state.xml`) every `--chunk-ns` (default `5 ns`), allowing interrupted runs to resume seamlessly across heterogeneous GPUs (e.g. local CUDA workstation and Google Colab L4/T4/A100) without binary `.chk` incompatibility.
- **Mechanistic & statistical rigour**: Evaluates post-burn-in statistical inefficiency ($g$) and effective sample size ($N_{\mathrm{eff}} = N_{\mathrm{eq}} / g$) via `pymbar`, block-averaged standard errors, all 4 ZAFF $\mathrm{Zn^{2+}}$ bond lengths, buried surface area, DSSP helicity (computed on intact chains prior to residue slicing), and GB-Neck2 (`igb=8`, `mbondi3`, `saltcon=0.15 M`) per-residue MM/GBSA decomposition.

## Repository Layout

```text
.
├── .github/workflows/ci.yml              # Automated syntax & unit-test CI
├── environment.yml                       # Conda/Mamba environment (OpenMM 8, AmberTools, PyMOL, ffmpeg)
├── notebooks/
│   └── colab_md_runner.ipynb             # Self-resuming Google Colab GPU runner notebook
├── params/
│   ├── ZAFF.frcmod                       # AMBER ZAFF bonded zinc force-field modifications (Peters et al., 2010)
│   └── ZAFF_H.prep                       # AMBER ZAFF residue prep definitions
├── scripts/
│   ├── 01_build_system.py                # PDBFixer repair/mutagenesis + ACE/NME capping + tleap + ParmEd HMR
│   ├── 02_run_md.py                      # Roe & Brooks 10-step relaxation + 4 fs HMR OpenMM production + solvent stripping
│   ├── 03_analyze_and_mmpbsa.py          # Multi-replica QC, mechanistic contacts, pymbar stats, plots & MM/GBSA
│   ├── 04_render_trajectory_movie.py     # Headless CPU PyMOL ray-traced trajectory movie + live metric HUD
│   └── run_pipeline.sh                   # Portable Bash wrapper for build + multi-replica MD execution
├── tests/
│   └── test_pipeline_utils.py            # Unit tests for mutation parsing, capping geometry, and statistical helpers
├── LICENSE
└── README.md
```

## Installation

Create the environment with `mamba` or `conda` on Linux (or WSL2 with NVIDIA CUDA drivers):

```bash
mamba env create -f environment.yml
mamba activate md-openmm
python -m openmm.testInstallation
```

> Match `cuda-version` to your host NVIDIA driver if required (e.g. `mamba install -c conda-forge cuda-version=12.4`).

## Usage

### 1. Build Solvated + HMR Topologies

```bash
# Wild-type complexes
python scripts/01_build_system.py --system 4XXB --mutations WT --variant WT --outdir runs/4XXB_WT
python scripts/01_build_system.py --system 2AC0_dimer --mutations WT --variant WT --outdir runs/2AC0_dimer_WT

# Point-mutation examples (arbitrary CHAIN:X123Y specifications)
python scripts/01_build_system.py --system 4XXB --mutations "B:R326H" --variant R326H --outdir runs/4XXB_R326H
python scripts/01_build_system.py --system 2AC0_dimer --mutations "A:R280K,B:R280K" --variant R280K_hom --outdir runs/2AC0_dimer_R280K_hom
```

### 2. Run Relaxation and Production MD

Run a single replica (resumes automatically from `state.xml` if interrupted):

```bash
python scripts/02_run_md.py \
  --run-dir runs/4XXB_WT \
  --replica 1 \
  --ns 200 \
  --chunk-ns 5 \
  --save-ps 20 \
  --dt-fs 4.0 \
  --platform CUDA
```

Or run build + multiple replicas sequentially via the portable shell wrapper:

```bash
bash scripts/run_pipeline.sh 4XXB WT WT 1,2,3
```

For cloud GPU execution, open `notebooks/colab_md_runner.ipynb` in Google Colab and point `DRIVE_PIPELINE_DIR` to your synced project folder.

### 3. Multi-Replica Analysis & MM/GBSA Decomposition

Runs on lightweight solvent-stripped `prod_dry.dcd` trajectories (suitable for CPU workstations):

```bash
python scripts/03_analyze_and_mmpbsa.py \
  --system 4XXB \
  --wt-dir runs/4XXB_WT \
  --mut-dir runs/4XXB_R326H \
  --replicas 1,2,3 \
  --outdir analysis/4XXB_WT_vs_R326H \
  --run-mmpbsa
```

Outputs written to `--outdir`:

- `*_replica_metrics.csv` and `*_summary_mean_sem.csv`: Per-replica QC metrics, `pymbar` statistical inefficiency ($g$, $N_{\mathrm{eff}}$, automated $t_0$ diagnostic), $\mathrm{Zn^{2+}}$ bond lengths, BSA, and mechanistic contacts.
- `*_WT_vs_*.png`: 300-DPI 3-panel comparison (backbone RMSD traces, per-residue $\mathrm{C\alpha}$ RMSF mean $\pm$ SD, and key contact distance distributions).
- `mmpbsa_*/FINAL_RESULTS_MMGBSA.dat` and `FINAL_DECOMP_MMGBSA.dat`: GB-Neck2 (`igb=8`, `mbondi3`, `saltcon=0.15 M`) binding free energies and per-residue decomposition.

### 4. Synchronised Trajectory Movie Rendering (4XXB, CPU-Only)

```bash
python scripts/04_render_trajectory_movie.py \
  --run-dir runs/4XXB_WT \
  --replica 1 \
  --outdir analysis/movies/4XXB_WT_rep1 \
  --n-frames 201 \
  --fps 15
```

Produces a multi-state PyMOL session (`.pse`), medoid representative PDB structure, H.264 `.mp4`, and `.gif` preview with a synchronised live metric HUD.

## Testing

```bash
python -m unittest discover -s tests -v
```

## References & Attribution

- **ZAFF parameters** (`params/ZAFF.frcmod`, `params/ZAFF_H.prep`): Peters, M. B. *et al.* (2010). Structural survey of zinc-containing proteins and development of the Zinc AMBER Force Field (ZAFF). *J. Chem. Theory Comput.* 6(9), 2935–2947.
- **Equilibration protocol**: Roe, D. R. & Brooks, B. R. (2020). A protocol for preparing explicitly solvated systems for stable molecular dynamics simulations. *J. Chem. Phys.* 153(5), 054123.
- **Hydrogen mass repartitioning**: Hopkins, C. W. *et al.* (2015). Long-time-step molecular dynamics through hydrogen mass repartitioning. *J. Chem. Theory Comput.* 11(4), 1864–1874.
- **Crystal structures**: RCSB PDB entries [`4XXB`](https://www.rcsb.org/structure/4XXB) (MDM2–RPL11) and [`2AC0`](https://www.rcsb.org/structure/2AC0) (p53 DBD–DNA).

## Licence

Pipeline scripts and documentation are released under the [MIT License](LICENSE). Third-party AMBER ZAFF parameter files in `params/` retain their original attribution and academic distribution terms.
