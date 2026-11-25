#!/usr/bin/env python3
"""
03_analyze_and_mmpbsa.py
Low-RAM MD trajectory analysis and MM/GBSA per-residue decomposition suite.
Compares wild-type vs mutant trajectories across independent replicas (rep1..repN) using:
  - Fixed burn-in truncation (default 20%) + pymbar statistical inefficiency (g, N_eff = N_eq / g)
    and automated detect_equilibration t0 diagnostic + block-averaged standard errors
  - Zn2+ tetrahedral coordination bond-length QC across all ZAFF bonds
  - System-specific mechanistic observables:
      * 4XXB       : MDM2 R326(Mut)-RPL11 Asn97 O distance/occupancy, R326-Trp303/Trp329 packing,
                     MDM2 displacement on RPL11-aligned frame, MDM2-RPL11 buried surface area (BSA)
      * 2AC0_dimer : p53 A/B R280(Mut)-DNA DG8 (O6/N7) H-bond occupancy, R280-Asp281 salt bridge,
                     Helix H2 (278-286) DSSP helicity (computed on intact chains), H1 dimer interface distance
  - Optional AmberTools MMPBSA.py GB-Neck2 (igb=8, mbondi3 radii, saltcon=0.15 M) per-residue decomposition.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_res_map(run_dir: Path) -> Tuple[dict, Dict[Tuple[str, int], int]]:
    meta = json.loads((run_dir / "build" / "residue_map.json").read_text(encoding="utf-8"))
    # Map (chain, pdb_resseq) -> 0-based mdtraj residue index (tleap_idx - 1)
    lookup = {(r["chain"], r["pdb_resseq"]): r["tleap_idx"] - 1 for r in meta["residues"]}
    return meta, lookup


def block_average_sem(x: np.ndarray, n_blocks: int = 5) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    if len(x) < n_blocks * 2:
        return float(np.std(x, ddof=1) / np.sqrt(len(x)))
    trim = (len(x) // n_blocks) * n_blocks
    blocks = x[:trim].reshape(n_blocks, -1).mean(axis=1)
    return float(np.std(blocks, ddof=1) / np.sqrt(n_blocks))


def min_dist_between_selections(traj, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    import mdtraj as md

    pairs = np.array([(i, j) for i in idx_a for j in idx_b], dtype=int)
    dists_nm = md.compute_distances(traj, pairs)
    return dists_nm.min(axis=1) * 10.0  # Angstroms


def centroid_distance(traj, idx_a: np.ndarray, idx_b: np.ndarray) -> np.ndarray:
    ca = traj.xyz[:, idx_a, :].mean(axis=1)
    cb = traj.xyz[:, idx_b, :].mean(axis=1)
    return np.linalg.norm(ca - cb, axis=1) * 10.0  # Angstroms


def sidechain_heavy(top, res_idx0: int) -> np.ndarray:
    bb = {"N", "CA", "C", "O", "H", "HA"}
    res = top.residue(res_idx0)
    idx = [a.index for a in res.atoms if a.element.symbol != "H" and a.name not in bb]
    if not idx:
        idx = [a.index for a in res.atoms if a.name == "CA"]
    return np.array(idx, dtype=int)


def trp_ring_atoms(top, res_idx0: int) -> np.ndarray:
    ring = {"CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"}
    res = top.residue(res_idx0)
    return np.array([a.index for a in res.atoms if a.name in ring], dtype=int)


def analyze_single_replica(run_dir: Path, rep_id: int, system_key: str, discard_frac: float = 0.2) -> dict:
    import mdtraj as md
    from pymbar import timeseries

    meta, lookup = load_res_map(run_dir)
    dry_prmtop = run_dir / "build" / "dry.prmtop"
    dry_dcd = run_dir / f"rep{rep_id}" / "prod_dry.dcd"
    if not dry_dcd.exists():
        raise FileNotFoundError(f"Missing {dry_dcd}. Finish 02_run_md.py first.")

    traj = md.load_dcd(str(dry_dcd), top=str(dry_prmtop))
    top = traj.topology
    mols = top.find_molecules()
    if len(mols) > 1:
        traj.image_molecules(inplace=True, anchor_molecules=[mols[0]])

    n_frames = traj.n_frames
    start = int(n_frames * discard_frac)
    eq_traj = traj[start:]

    # 1. Global backbone RMSD (aligned to frame 0)
    ca_idx = top.select("name CA")
    traj.superpose(traj, 0, atom_indices=ca_idx)
    rmsd_ang = md.rmsd(traj, traj, 0, atom_indices=ca_idx) * 10.0
    rmsd_eq = rmsd_ang[start:]

    # Compute statistical inefficiency directly on the retained post-burn-in window so N_eff == N_eq / g,
    # while recording pymbar's automated t0 separately as an equilibration convergence diagnostic.
    g = float(timeseries.statistical_inefficiency(rmsd_eq))
    n_eff = float(len(rmsd_eq) / max(g, 1.0))
    auto_t0_rel, _, _ = timeseries.detect_equilibration(rmsd_eq)

    # 2. Zn2+ tetrahedral coordination QC (all ZAFF bonds from residue_map.json)
    zn_bonds = {}
    for b in meta["bonds"]:
        z_res = top.residue(b["zn_idx"] - 1)
        l_res = top.residue(b["lig_idx"] - 1)
        z_at = [a.index for a in z_res.atoms if a.name == "ZN"][0]
        l_at = [a.index for a in l_res.atoms if a.name == b["lig_atom"]][0]
        d_ang = md.compute_distances(eq_traj, [[z_at, l_at]])[:, 0] * 10.0
        zn_bonds[f"ZN{b['zn_idx']}-{l_res.name}{b['lig_idx']}.{b['lig_atom']}"] = float(d_ang.mean())

    metrics = {
        "replica": rep_id,
        "n_frames_total": n_frames,
        "n_frames_eq": eq_traj.n_frames,
        "pymbar_auto_t0_frame": start + int(auto_t0_rel),
        "stat_inefficiency_g": round(g, 2),
        "n_eff": round(n_eff, 1),
        "backbone_rmsd_mean_A": float(rmsd_eq.mean()),
        "backbone_rmsd_sem_A": block_average_sem(rmsd_eq),
        "zn_bond_max_mean_A": max(zn_bonds.values()),
        "zn_bond_min_mean_A": min(zn_bonds.values()),
    }

    series = {"rmsd_A": rmsd_ang}
    rmsf_A = md.rmsf(eq_traj, eq_traj, 0, atom_indices=ca_idx) * 10.0
    series["ca_rmsf_A"] = rmsf_A

    if system_key == "4XXB":
        r326_idx = lookup[("B", 326)]
        asn97_idx = lookup[("A", 97)]
        w303_idx = lookup[("B", 303)]
        w329_idx = lookup[("B", 329)]

        r326_sc = sidechain_heavy(top, r326_idx)
        asn97_o = np.array([a.index for a in top.residue(asn97_idx).atoms if a.name == "O"], dtype=int)
        w303_ring = trp_ring_atoms(top, w303_idx)
        w329_ring = trp_ring_atoms(top, w329_idx)

        d_r326_asn97 = min_dist_between_selections(eq_traj, r326_sc, asn97_o)
        d_r326_w303 = centroid_distance(eq_traj, r326_sc, w303_ring)
        d_r326_w329 = centroid_distance(eq_traj, r326_sc, w329_ring)

        idx_rpl11 = np.array([a.index for a in top.atoms if meta["residues"][a.residue.index]["chain"] == "A"])
        idx_mdm2 = np.array([a.index for a in top.atoms if meta["residues"][a.residue.index]["chain"] == "B"])
        sub = eq_traj[:: max(1, eq_traj.n_frames // 200)]
        sasa_ab = md.shrake_rupley(sub, mode="residue").sum(axis=1)
        sasa_a = md.shrake_rupley(sub.atom_slice(idx_rpl11), mode="residue").sum(axis=1)
        sasa_b = md.shrake_rupley(sub.atom_slice(idx_mdm2), mode="residue").sum(axis=1)
        bsa_A2 = 0.5 * (sasa_a + sasa_b - sasa_ab) * 100.0  # nm^2 -> A^2

        rpl11_ca = [a.index for a in top.atoms if a.name == "CA" and meta["residues"][a.residue.index]["chain"] == "A"]
        mdm2_ca = [a.index for a in top.atoms if a.name == "CA" and meta["residues"][a.residue.index]["chain"] == "B"]
        aligned = eq_traj.slice(range(eq_traj.n_frames), copy=True)
        aligned.superpose(traj, 0, atom_indices=rpl11_ca)
        diff_nm = aligned.xyz[:, mdm2_ca, :] - traj.xyz[0, mdm2_ca, :]
        mdm2_disp_A = np.sqrt((diff_nm ** 2).sum(axis=-1).mean(axis=-1)) * 10.0

        metrics.update({
            "R326_Asn97O_dist_mean_A": float(d_r326_asn97.mean()),
            "R326_Asn97O_occ_lt3p5A": float((d_r326_asn97 < 3.5).mean()),
            "R326_Trp303_centroid_A": float(d_r326_w303.mean()),
            "R326_Trp329_centroid_A": float(d_r326_w329.mean()),
            "MDM2_on_RPL11_RMSD_A": float(mdm2_disp_A.mean()),
            "BSA_MDM2_RPL11_A2": float(bsa_A2.mean()),
        })
        series["key_contact_A"] = d_r326_asn97

    elif system_key == "2AC0_dimer":
        # Compute DSSP on the intact solute trajectory once so flanking residues preserve H-bond context
        sub_dssp = eq_traj[:: max(1, eq_traj.n_frames // 200)]
        full_dssp = md.compute_dssp(sub_dssp, simplified=True)

        occ_list, dist_list, sb_list, helicity_list, all_d_dna = [], [], [], [], []
        for pch, dch in [("A", "F"), ("B", "E")]:
            r280_sc = sidechain_heavy(top, lookup[(pch, 280)])
            d281_od = np.array([a.index for a in top.residue(lookup[(pch, 281)]).atoms if a.name in ("OD1", "OD2")])
            dg8_acc = np.array([a.index for a in top.residue(lookup[(dch, 8)]).atoms if a.name in ("O6", "N7")])

            d_dna = min_dist_between_selections(eq_traj, r280_sc, dg8_acc)
            d_sb = min_dist_between_selections(eq_traj, r280_sc, d281_od)
            all_d_dna.append(d_dna)
            dist_list.append(float(d_dna.mean()))
            occ_list.append(float((d_dna < 3.5).mean()))
            sb_list.append(float((d_sb < 3.5).mean()))

            h2_res = [lookup[(pch, r)] for r in range(278, 287)]
            helicity_list.append(float((full_dssp[:, h2_res] == "H").mean()))

        h1_a = np.array([a.index for r in range(177, 182) for a in top.residue(lookup[("A", r)]).atoms if a.name == "CA"])
        h1_b = np.array([a.index for r in range(177, 182) for a in top.residue(lookup[("B", r)]).atoms if a.name == "CA"])
        d_h1 = centroid_distance(eq_traj, h1_a, h1_b)

        metrics.update({
            "R280_DG8_dist_mean_A": float(np.mean(dist_list)),
            "R280_DG8_Hbond_occ_lt3p5A": float(np.mean(occ_list)),
            "R280_Asp281_saltbridge_occ": float(np.mean(sb_list)),
            "HelixH2_helicity_frac": float(np.mean(helicity_list)),
            "H1_dimer_interface_A": float(d_h1.mean()),
        })
        series["key_contact_A"] = np.concatenate(all_d_dna)

    return {"metrics": metrics, "series": series, "zn_bonds": zn_bonds}


def run_mmpbsa_for_variant(
    run_dir: Path,
    system_key: str,
    replicas: List[int],
    copy_results_to: Path | None = None,
) -> Path | None:
    if shutil.which("MMPBSA.py") is None or shutil.which("ante-MMPBSA.py") is None:
        print("[WARN] MMPBSA.py / ante-MMPBSA.py not found in PATH; skipping MM/GBSA.", flush=True)
        return None

    import mdtraj as md
    import parmed as pmd

    meta, _ = load_res_map(run_dir)
    mmpbsa_dir = (run_dir / f"rep{replicas[0]}" / "mmpbsa") if len(replicas) == 1 else (run_dir / "mmpbsa")
    mmpbsa_dir.mkdir(parents=True, exist_ok=True)

    if system_key == "4XXB":
        lig_res = [r["tleap_idx"] for r in meta["residues"] if r["chain"] == "B" or r["role"] == "zn"]
    else:
        lig_res = [r["tleap_idx"] for r in meta["residues"] if r["role"] == "dna"]

    lig_mask = f":{min(lig_res)}-{max(lig_res)}"
    dry_prmtop = run_dir / "build" / "dry.prmtop"

    # Generate complex.prmtop with mbondi3 radii directly for igb=8
    c_parm = pmd.load_file(str(dry_prmtop))
    pmd.tools.changeRadii(c_parm, "mbondi3").execute()
    c_parm.save(str(mmpbsa_dir / "complex.prmtop"), overwrite=True)

    for old_top in ("receptor.prmtop", "ligand.prmtop"):
        old_p = mmpbsa_dir / old_top
        if old_p.exists():
            old_p.unlink()

    subprocess.run(
        [
            "ante-MMPBSA.py",
            "-p", "complex.prmtop",
            "-r", "receptor.prmtop",
            "-l", "ligand.prmtop",
            "-n", lig_mask,
            "--radii=mbondi3",
        ],
        cwd=str(mmpbsa_dir),
        check=True,
    )

    trajin_lines = []
    for r in replicas:
        dcd = run_dir / f"rep{r}" / "prod_dry.dcd"
        t = md.load_dcd(str(dcd), top=str(dry_prmtop))
        start = int(t.n_frames * 0.2) + 1
        stride = max(1, (t.n_frames - start) // max(1, 200 // len(replicas)))
        rel_dcd = os.path.relpath(dcd, mmpbsa_dir)
        trajin_lines.append(f"trajin {rel_dcd} {start} last {stride}")
    cpp_in = mmpbsa_dir / "pool.cpptraj"
    cpp_in.write_text(
        "\n".join(trajin_lines) + "\nautoimage\ntrajout pooled_eq.nc netcdf\nrun\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["cpptraj", "-p", os.path.relpath(dry_prmtop, mmpbsa_dir), "-i", cpp_in.name],
        cwd=str(mmpbsa_dir),
        check=True,
    )

    mmpbsa_in = mmpbsa_dir / "mmgbsa.in"
    mmpbsa_in.write_text(
        """&general
  startframe=1, endframe=999999, interval=1,
  verbose=1, keep_files=0, netcdf=1,
/
&gb
  igb=8, saltcon=0.15,
/
&decomposition
  idecomp=1, dec_verbose=1,
/
""",
        encoding="utf-8",
    )
    out_dat = mmpbsa_dir / "FINAL_RESULTS_MMGBSA.dat"
    out_decomp = mmpbsa_dir / "FINAL_DECOMP_MMGBSA.dat"
    subprocess.run(
        [
            "MMPBSA.py", "-O",
            "-i", mmpbsa_in.name,
            "-o", out_dat.name,
            "-do", out_decomp.name,
            "-cp", "complex.prmtop",
            "-rp", "receptor.prmtop",
            "-lp", "ligand.prmtop",
            "-y", "pooled_eq.nc",
        ],
        cwd=str(mmpbsa_dir),
        check=True,
    )
    if copy_results_to is not None:
        copy_results_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_dat, copy_results_to / out_dat.name)
        shutil.copy2(out_decomp, copy_results_to / out_decomp.name)
    print(f"[OK] MM/GBSA complete -> {out_dat}", flush=True)
    return out_dat


def plot_comparison(
    wt_res: List[dict],
    mut_res: List[dict],
    system_key: str,
    out_png: Path,
    wt_label: str,
    mut_label: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=300)

    # Panel A: Backbone RMSD traces
    for i, r in enumerate(wt_res):
        axes[0].plot(r["series"]["rmsd_A"], color="#1f77b4", alpha=0.55, lw=1.0, label=wt_label if i == 0 else None)
    for i, r in enumerate(mut_res):
        axes[0].plot(r["series"]["rmsd_A"], color="#d62728", alpha=0.55, lw=1.0, label=mut_label if i == 0 else None)
    axes[0].set_title(f"{system_key}: Backbone RMSD")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("RMSD (Å)")
    axes[0].legend(frameon=False)

    # Panel B: Per-residue C-alpha RMSF (mean +/- sample SD across replicas)
    wt_rmsf = np.vstack([r["series"]["ca_rmsf_A"] for r in wt_res])
    mut_rmsf = np.vstack([r["series"]["ca_rmsf_A"] for r in mut_res])
    x = np.arange(1, wt_rmsf.shape[1] + 1)
    wt_sd = wt_rmsf.std(axis=0, ddof=1) if wt_rmsf.shape[0] >= 2 else np.zeros_like(wt_rmsf[0])
    mut_sd = mut_rmsf.std(axis=0, ddof=1) if mut_rmsf.shape[0] >= 2 else np.zeros_like(mut_rmsf[0])

    axes[1].plot(x, wt_rmsf.mean(0), color="#1f77b4", label=wt_label)
    axes[1].fill_between(x, wt_rmsf.mean(0) - wt_sd, wt_rmsf.mean(0) + wt_sd, color="#1f77b4", alpha=0.2)
    axes[1].plot(x, mut_rmsf.mean(0), color="#d62728", label=mut_label)
    axes[1].fill_between(x, mut_rmsf.mean(0) - mut_sd, mut_rmsf.mean(0) + mut_sd, color="#d62728", alpha=0.2)
    axes[1].set_title("Cα RMSF (mean ± SD)")
    axes[1].set_xlabel("Sequential Cα Index")
    axes[1].set_ylabel("RMSF (Å)")
    axes[1].legend(frameon=False)

    # Panel C: Key mechanistic contact distance distribution
    wt_d = np.concatenate([r["series"]["key_contact_A"] for r in wt_res])
    mut_d = np.concatenate([r["series"]["key_contact_A"] for r in mut_res])
    axes[2].hist(wt_d, bins=40, density=True, alpha=0.55, color="#1f77b4", label=wt_label)
    axes[2].hist(mut_d, bins=40, density=True, alpha=0.55, color="#d62728", label=mut_label)
    axes[2].axvline(3.5, ls="--", color="k", lw=0.9, label="H-bond 3.5 Å")
    title_c = "MDM2 326 – RPL11 Asn97 O" if system_key == "4XXB" else "p53 280 – DNA DG8 (O6/N7)"
    axes[2].set_title(title_c)
    axes[2].set_xlabel("Minimum Distance (Å)")
    axes[2].set_ylabel("Probability Density")
    axes[2].legend(frameon=False)

    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)
    print(f"[OK] Saved comparison figure -> {out_png}")


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description="Analyse WT vs Mutant MD trajectories and optionally run MM/GBSA.")
    ap.add_argument("--system", required=True, choices=["4XXB", "2AC0_dimer"])
    ap.add_argument("--wt-dir", required=True, help="Primary (e.g. WT) run directory.")
    ap.add_argument("--variant-label", default=None, help="Label for --wt-dir (default: 'WT' when --mut-dir is given, else directory name).")
    ap.add_argument("--mut-dir", default=None, help="Optional Mutant run directory for head-to-head comparison.")
    ap.add_argument("--replicas", default="1,2,3", help="Comma-separated replica indices (default: 1,2,3).")
    ap.add_argument("--discard-frac", type=float, default=0.2, help="Initial burn-in fraction discarded before statistics (default: 0.2).")
    ap.add_argument("--outdir", required=True, help="Directory for summary CSVs and plots.")
    ap.add_argument("--run-mmpbsa", action="store_true", help="Run MMPBSA.py GB-Neck2 (igb=8) + per-residue decomposition.")
    args = ap.parse_args()

    reps = [int(x.strip()) for x in args.replicas.split(",") if x.strip()]
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    wt_dir = Path(args.wt_dir).resolve()
    wt_label = args.variant_label or ("WT" if args.mut_dir else wt_dir.name)
    wt_res = [analyze_single_replica(wt_dir, r, args.system, discard_frac=args.discard_frac) for r in reps]
    rows = [{"variant": wt_label, **x["metrics"]} for x in wt_res]

    if args.mut_dir:
        mut_dir = Path(args.mut_dir).resolve()
        mut_label = mut_dir.name
        mut_res = [analyze_single_replica(mut_dir, r, args.system, discard_frac=args.discard_frac) for r in reps]
        rows.extend([{"variant": mut_label, **x["metrics"]} for x in mut_res])
        plot_comparison(
            wt_res, mut_res, args.system,
            outdir / f"{args.system}_{wt_label}_vs_{mut_label}.png",
            wt_label=wt_label, mut_label=mut_label,
        )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"{args.system}_replica_metrics.csv", index=False)

    agg_cols = [c for c in df.columns if c not in ("variant", "replica")]
    summary = df.groupby("variant")[agg_cols].agg(["mean", "sem"])
    summary.to_csv(outdir / f"{args.system}_summary_mean_sem.csv")
    print(f"\n=== {args.system} SUMMARY (mean ± SEM across {len(reps)} replicas) ===")
    print(summary.to_string())

    if args.run_mmpbsa:
        run_mmpbsa_for_variant(
            wt_dir, args.system, reps,
            copy_results_to=outdir / ("mmpbsa" if not args.mut_dir else f"mmpbsa_{wt_dir.name}"),
        )
        if args.mut_dir:
            mut_path = Path(args.mut_dir).resolve()
            run_mmpbsa_for_variant(
                mut_path, args.system, reps,
                copy_results_to=outdir / f"mmpbsa_{mut_path.name}",
            )


if __name__ == "__main__":
    main()
