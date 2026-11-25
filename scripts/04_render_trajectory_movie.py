#!/usr/bin/env python3
"""
04_render_trajectory_movie.py
CPU-only publication-grade 4XXB MD trajectory animation builder using PyMOL (headless software ray-tracing)
+ Matplotlib synchronised live metric panels + Pillow HUD + ffmpeg H.264 MP4 & GIF encoder.

Strictly zero GPU usage (CUDA_VISIBLE_DEVICES="" + PYOPENGL_PLATFORM=osmesa).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Force strictly CPU-only headless execution before importing visualisation libraries
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import numpy as np


def load_res_map(run_dir: Path) -> Tuple[dict, Dict[Tuple[str, int], int]]:
    meta = json.loads((run_dir / "build" / "residue_map.json").read_text(encoding="utf-8"))
    lookup = {(r["chain"], r["pdb_resseq"]): r["tleap_idx"] - 1 for r in meta["residues"]}
    return meta, lookup


def write_multimodel_pdb(
    traj,
    meta: dict,
    out_pdb: Path,
    frame_indices: np.ndarray | List[int],
) -> None:
    """
    Write a clean multi-model PDB with canonical PDB chain IDs and residue numbers.
    Renames ZAFF CY1/CY2 -> CYS and HE1 -> HIS, ZN1/ZN2 -> ZN so PyMOL draws continuous
    ribbons through zinc-coordinating residues and recognises ZN as a metal ion.
    """
    res_info = meta["residues"]
    top = traj.topology
    rename_map = {"CY1": "CYS", "CY2": "CYS", "HE1": "HIS", "ZN1": "ZN", "ZN2": "ZN"}

    atom_meta: List[Tuple[str, int, str, str, str, int, str]] = []
    res_chain_prev = None
    ter_before_atom: set[int] = set()
    for atom in top.atoms:
        r_idx = atom.residue.index
        rm = res_info[r_idx]
        role = rm["role"]
        if role == "cap":
            continue
        elem = atom.element.symbol if atom.element is not None else "C"
        if elem == "H":
            continue
        ch = "Z" if role == "zn" else rm["chain"]
        if res_chain_prev is not None and ch != res_chain_prev:
            ter_before_atom.add(atom.index)
        res_chain_prev = ch
        rec = "HETATM" if role == "zn" else "ATOM"
        rn = rename_map.get(rm["resname"], rm["resname"])
        resseq = int(rm["pdb_resseq"])
        an = f" {atom.name:<3s}" if len(atom.name) < 4 else f"{atom.name:<4s}"
        atom_meta.append((rec, atom.index, an, rn, ch, resseq, elem))

    lines: List[str] = []
    for m_idx, f_idx in enumerate(frame_indices, start=1):
        xyz_A = traj.xyz[int(f_idx)] * 10.0
        lines.append(f"MODEL     {m_idx:4d}")
        s = 1
        for rec, a_idx, an, rn, ch, resseq, elem in atom_meta:
            if a_idx in ter_before_atom:
                lines.append("TER")
            x, y, z = xyz_A[a_idx]
            lines.append(
                f"{rec:<6s}{s:5d} {an:4s} {rn:>3s} {ch:1s}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}"
            )
            s += 1
        lines.append("TER")
        lines.append("ENDMDL")
    lines.append("END")
    out_pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_4xxb_data(
    run_dir: Path,
    replica: int,
    outdir: Path,
    n_movie_frames: int,
    save_ps: float = 20.0,
) -> dict:
    import mdtraj as md

    meta, lookup = load_res_map(run_dir)
    if meta.get("system_key", meta.get("system")) != "4XXB":
        raise ValueError(
            f"04_render_trajectory_movie.py currently supports system '4XXB' only (got '{meta.get('system_key')}')."
        )

    dry_prmtop = run_dir / "build" / "dry.prmtop"
    dry_dcd = run_dir / f"rep{replica}" / "prod_dry.dcd"

    print(f"[STEP 1] Loading {dry_dcd} ...", flush=True)
    traj = md.load_dcd(str(dry_dcd), top=str(dry_prmtop))
    top = traj.topology
    mols = top.find_molecules()
    if len(mols) > 1:
        print(f"[STEP 1] PBC imaging {len(mols)} molecules onto anchor molecule 0 ...", flush=True)
        traj.image_molecules(inplace=True, anchor_molecules=[mols[0]])

    n_total = traj.n_frames
    step_ns = save_ps / 1000.0
    times_ns = np.linspace(step_ns, n_total * step_ns, n_total)

    ca_idx = top.select("name CA")
    rmsd_all_A = md.rmsd(traj, traj, 0, atom_indices=ca_idx) * 10.0

    r326_idx = lookup[("B", 326)]
    asn97_idx = lookup[("A", 97)]
    bb = {"N", "CA", "C", "O", "H", "HA"}
    r326_sc = np.array(
        [a.index for a in top.residue(r326_idx).atoms if a.element.symbol != "H" and a.name not in bb], dtype=int
    )
    asn97_o = np.array([a.index for a in top.residue(asn97_idx).atoms if a.name == "O"], dtype=int)
    pairs_r326 = np.array([(i, j) for i in r326_sc for j in asn97_o], dtype=int)
    d_r326_asn97_A = md.compute_distances(traj, pairs_r326).min(axis=1) * 10.0

    zn_pairs = []
    for b in meta["bonds"]:
        z_at = [a.index for a in top.residue(b["zn_idx"] - 1).atoms if a.name == "ZN"][0]
        l_at = [a.index for a in top.residue(b["lig_idx"] - 1).atoms if a.name == b["lig_atom"]][0]
        zn_pairs.append([z_at, l_at])
    zn_dists_A = md.compute_distances(traj, zn_pairs) * 10.0

    # Align trajectory smoothly on rigid RPL11 core CA (13-138, 154-178; excluding rebuilt loop 139-153)
    core_ca = [
        a.index
        for a in top.atoms
        if a.name == "CA"
        and meta["residues"][a.residue.index]["chain"] == "A"
        and not (139 <= meta["residues"][a.residue.index]["pdb_resseq"] <= 153)
    ]
    print(f"[STEP 1] Aligning {n_total} frames on {len(core_ca)} rigid RPL11 core CA atoms ...", flush=True)
    traj.superpose(traj, 0, atom_indices=core_ca)

    frame_indices = np.unique(np.round(np.linspace(0, n_total - 1, n_movie_frames)).astype(int))
    movie_pdb = outdir / "traj_movie.pdb"
    print(f"[STEP 1] Writing {len(frame_indices)}-frame PDB -> {movie_pdb} ...", flush=True)
    write_multimodel_pdb(traj, meta, movie_pdb, frame_indices)

    eq_start = int(n_total * 0.2)
    eq_xyz = traj.xyz[eq_start:, ca_idx, :]
    mean_xyz = eq_xyz.mean(axis=0)
    dist_to_mean = np.sqrt(((eq_xyz - mean_xyz) ** 2).sum(axis=-1).mean(axis=-1))
    rep_frame_idx = eq_start + int(np.argmin(dist_to_mean))
    rep_pdb = outdir / f"{run_dir.name}_rep{replica}_representative.pdb"
    write_multimodel_pdb(traj, meta, rep_pdb, [rep_frame_idx])
    print(
        f"[STEP 1] Saved representative equilibrated frame (t={times_ns[rep_frame_idx]:.2f} ns, idx={rep_frame_idx}) -> {rep_pdb}",
        flush=True,
    )

    metrics = {
        "times_ns": times_ns,
        "rmsd_A": rmsd_all_A,
        "d_r326_asn97_A": d_r326_asn97_A,
        "zn_dists_A": zn_dists_A,
        "frame_indices": frame_indices,
        "movie_times_ns": times_ns[frame_indices],
        "rep_frame_time_ns": float(times_ns[rep_frame_idx]),
        "movie_pdb": str(movie_pdb),
        "rep_pdb": str(rep_pdb),
    }
    np.savez(
        outdir / "movie_metrics.npz",
        times_ns=times_ns,
        rmsd_A=rmsd_all_A,
        d_r326_asn97_A=d_r326_asn97_A,
        zn_dists_A=zn_dists_A,
        frame_indices=frame_indices,
        movie_times_ns=times_ns[frame_indices],
    )
    return metrics


def render_pymol_frames(
    movie_pdb: Path,
    frames_dir: Path,
    pse_path: Path,
    n_states: int,
    width: int = 960,
    height: int = 720,
    max_threads: int = 4,
    preview_only: bool = False,
) -> None:
    import pymol  # type: ignore

    pymol.pymol_argv = ["pymol", "-cq"]
    pymol.finish_launching()
    from pymol import cmd  # type: ignore

    cmd.set("max_threads", max_threads)
    cmd.load(str(movie_pdb), "complex")
    n_atoms = cmd.count_atoms("all")
    if n_atoms == 0:
        print("[ERROR] PyMOL failed to load atoms!", flush=True)
        cmd.quit()
        sys.exit(1)
    print(f"[STEP 2] PyMOL loaded {n_atoms} atoms across {cmd.count_states('complex')} states.", flush=True)

    cmd.dss("complex", state=1)
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("ray_trace_mode", 0)
    cmd.set("ray_shadows", 0)
    cmd.set("depth_cue", 0)
    cmd.set("specular", 0.28)
    cmd.set("shininess", 25.0)
    cmd.set("ambient", 0.35)
    cmd.set("direct", 0.65)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_smooth_loops", 1)
    cmd.set("cartoon_flat_sheets", 1)
    cmd.set("cartoon_oval_width", 0.28)
    cmd.set("cartoon_oval_length", 1.30)
    cmd.set("stick_radius", 0.20)
    cmd.set("dash_gap", 0.22)
    cmd.set("dash_length", 0.26)
    cmd.set("dash_radius", 0.09)
    cmd.set("dash_color", "orange")
    cmd.set("orthoscopic", 1)

    cmd.hide("everything", "all")
    cmd.show("cartoon", "chain A or chain B")

    cmd.set_color("rpl11_blue", [0.30, 0.50, 0.72])
    cmd.set_color("rpl11_loop", [0.56, 0.76, 0.92])
    cmd.set_color("mdm2_coral", [0.89, 0.41, 0.31])
    cmd.set_color("r326_red", [0.88, 0.06, 0.16])
    cmd.set_color("trp_gold", [0.94, 0.62, 0.18])
    cmd.set_color("asn97_teal", [0.00, 0.66, 0.58])
    cmd.set_color("cys_yellow", [0.86, 0.72, 0.12])
    cmd.set_color("zn_violet", [0.48, 0.34, 0.88])

    cmd.color("rpl11_blue", "chain A")
    cmd.color("rpl11_loop", "chain A and resi 139-153")
    cmd.color("mdm2_coral", "chain B")

    cmd.select("zn_ion", "resn ZN")
    cmd.select("zn_cys", "chain B and resi 305+308+319+322")
    cmd.show("spheres", "zn_ion")
    cmd.set("sphere_scale", 0.85, "zn_ion")
    cmd.color("zn_violet", "zn_ion")
    cmd.show("sticks", "zn_cys and (sidechain or name CA)")
    cmd.color("cys_yellow", "zn_cys and elem C")
    cmd.util.cnc("zn_cys")
    cmd.distance("zn_bonds", "zn_ion", "zn_cys and name SG", cutoff=3.5)
    cmd.hide("labels", "zn_bonds")
    cmd.set("dash_color", "zn_violet", "zn_bonds")
    cmd.set("dash_radius", 0.07, "zn_bonds")

    cmd.select("r326", "chain B and resi 326")
    cmd.select("trp_pair", "chain B and resi 303+329")
    cmd.select("asn97", "chain A and resi 97")

    cmd.show("sticks", "trp_pair and (sidechain or name CA)")
    cmd.color("trp_gold", "trp_pair and elem C")
    cmd.util.cnc("trp_pair")

    cmd.show("sticks", "asn97 and (backbone or sidechain)")
    cmd.color("asn97_teal", "asn97 and elem C")
    cmd.util.cnc("asn97")

    cmd.show("sticks", "r326 and (sidechain or name CA)")
    cmd.set("stick_radius", 0.25, "r326")
    cmd.color("r326_red", "r326 and elem C")
    cmd.util.cnc("r326")

    cmd.distance("r326_asn97", "r326 and sidechain", "asn97 and name O", mode=0, cutoff=5.0)
    cmd.hide("labels", "r326_asn97")
    cmd.set("dash_color", "orange", "r326_asn97")
    cmd.set("dash_radius", 0.10, "r326_asn97")

    cmd.frame(1)
    cmd.orient("complex")
    view = list(cmd.get_view())
    rot = np.array(view[0:9]).reshape(3, 3)
    com_all = np.array(cmd.centerofmass("complex"))
    com_r326 = np.array(cmd.centerofmass("r326"))
    cam_z_in_model = rot[:, 2]
    if np.dot(com_r326 - com_all, cam_z_in_model) < 0:
        cmd.turn("y", 180)
    cmd.turn("x", 12)
    cmd.turn("y", -15)
    cmd.zoom("complex", buffer=0.5, state=1, complete=1)

    cmd.save(str(pse_path))
    print(f"[STEP 2] Saved multi-state PyMOL session -> {pse_path}", flush=True)

    frames_dir.mkdir(parents=True, exist_ok=True)
    end_state = 1 if preview_only else n_states
    for st in range(1, end_state + 1):
        cmd.frame(st)
        out_png = frames_dir / f"raw_{st:04d}.png"
        cmd.png(str(out_png), width=width, height=height, dpi=150, ray=1, quiet=1)
        if st == 1 or st % 25 == 0 or st == end_state:
            print(f"    [PyMOL] Rendered frame {st}/{end_state}", flush=True)

    cmd.quit()


def build_base_plot_panel(metrics: dict, label: str, panel_w: int = 520, panel_h: int = 720):
    """Render static right-hand 3-panel plot once, returning PIL Image + exact pixel bbox mapping."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    dpi = 100
    fig, axes = plt.subplots(3, 1, figsize=(panel_w / dpi, panel_h / dpi), dpi=dpi, sharex=True)
    fig.patch.set_facecolor("#F8FAFC")

    t = metrics["times_ns"]
    total_ns = float(t[-1])
    rmsd = metrics["rmsd_A"]
    d_r326 = metrics["d_r326_asn97_A"]
    zn = metrics["zn_dists_A"]

    def smooth(arr: np.ndarray, win: int = 31) -> np.ndarray:
        win = min(win, max(1, len(arr) if len(arr) % 2 == 1 else len(arr) - 1))
        pad = win // 2
        padded = np.pad(arr, (pad, pad), mode="edge")
        k = np.ones(win) / win
        return np.convolve(padded, k, mode="valid")

    ax0 = axes[0]
    ax0.set_facecolor("white")
    ax0.plot(t, rmsd, color="#3B82F6", alpha=0.22, lw=0.6)
    ax0.plot(t, smooth(rmsd, 31), color="#1D4ED8", lw=1.6, label="Backbone Cα RMSD")
    ax0.axhline(rmsd[int(len(rmsd) * 0.2):].mean(), color="#1E40AF", ls="--", lw=0.9, alpha=0.7)
    ax0.set_ylim(0, max(4.8, float(rmsd.max()) * 1.1))
    ax0.set_ylabel("RMSD (Å)", fontsize=10, fontweight="bold", color="#1E293B")
    ax0.set_title(f"{label} — Synchronised Trajectory QC", fontsize=11, fontweight="bold", color="#0F172A", pad=8)
    ax0.grid(alpha=0.25, ls=":")
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    ax1 = axes[1]
    ax1.set_facecolor("white")
    ax1.plot(t, d_r326, color="#EF4444", alpha=0.20, lw=0.6)
    ax1.plot(t, smooth(d_r326, 31), color="#DC2626", lw=1.6, label="Pos 326 – Asn97 O")
    ax1.axhline(3.5, color="#F59E0B", ls="--", lw=1.1, label="H-bond 3.5 Å")
    ax1.set_ylim(2.0, max(8.5, float(d_r326.max()) * 1.05))
    ax1.set_ylabel("326–N97 O (Å)", fontsize=10, fontweight="bold", color="#1E293B")
    ax1.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white", framealpha=0.85)
    ax1.grid(alpha=0.25, ls=":")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = axes[2]
    ax2.set_facecolor("white")
    zn_colors = ["#6D28D9", "#7C3AED", "#8B5CF6", "#A78BFA"]
    for i in range(zn.shape[1]):
        ax2.plot(t, smooth(zn[:, i], 31), color=zn_colors[i % len(zn_colors)], lw=1.2, alpha=0.85)
    ax2.axhline(2.426, color="#4C1D95", ls="--", lw=0.9, alpha=0.7, label="ZAFF req 2.43 Å")
    ax2.set_ylim(2.0, 2.9)
    ax2.set_xlim(0, total_ns)
    ax2.set_ylabel("4× Zn²⁺–SG (Å)", fontsize=10, fontweight="bold", color="#1E293B")
    ax2.set_xlabel("Simulation Time (ns)", fontsize=10, fontweight="bold", color="#1E293B")
    ax2.legend(loc="upper right", fontsize=8, frameon=True, facecolor="white", framealpha=0.85)
    ax2.grid(alpha=0.25, ls=":")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.subplots_adjust(left=0.16, right=0.95, top=0.93, bottom=0.09, hspace=0.22)
    fig.canvas.draw()

    bboxes = []
    for ax in axes:
        bb = ax.get_window_extent()
        bboxes.append({
            "x0": float(bb.x0),
            "x1": float(bb.x1),
            "y_top": float(panel_h - bb.y1),
            "y_bot": float(panel_h - bb.y0),
            "ymin": float(ax.get_ylim()[0]),
            "ymax": float(ax.get_ylim()[1]),
        })

    buf = np.asarray(fig.canvas.buffer_rgba())
    base_img = Image.fromarray(buf[:, :, :3].copy(), mode="RGB")
    plt.close(fig)
    return base_img, {"bboxes": bboxes, "xmin": 0.0, "xmax": total_ns}


def load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    return ImageFont.load_default()


def composite_frames(
    metrics: dict,
    frames_dir: Path,
    comp_dir: Path,
    label: str,
    preview_only: bool = False,
) -> None:
    from PIL import Image, ImageDraw

    comp_dir.mkdir(parents=True, exist_ok=True)
    panel_w, panel_h = 520, 720
    base_plot, plot_meta = build_base_plot_panel(metrics, label=label, panel_w=panel_w, panel_h=panel_h)
    total_ns = plot_meta["xmax"]

    font_title = load_font(22, bold=True)
    font_sub = load_font(15, bold=False)
    font_legend = load_font(14, bold=True)
    font_readout = load_font(13, bold=True)

    frame_indices = metrics["frame_indices"]
    n_states = 1 if preview_only else len(frame_indices)

    legend_items = [
        ((82, 128, 176), "RPL11 (A)"),
        ((222, 107, 82), "MDM2 (B)"),
        ((219, 20, 46), "Pos 326"),
        ((235, 158, 56), "W303/W329"),
        ((0, 166, 148), "Asn97"),
        ((117, 92, 217), "Zn²⁺-Cys₄"),
    ]

    for st in range(1, n_states + 1):
        raw_img = Image.open(frames_dir / f"raw_{st:04d}.png").convert("RGB")
        lw, lh = raw_img.size
        canvas = Image.new("RGB", (lw + panel_w, lh), (255, 255, 255))
        canvas.paste(raw_img, (0, 0))
        canvas.paste(base_plot, (lw, 0))

        draw = ImageDraw.Draw(canvas)
        f_idx = int(frame_indices[st - 1])
        t_ns = float(metrics["times_ns"][f_idx])
        rmsd_val = float(metrics["rmsd_A"][f_idx])
        r326_val = float(metrics["d_r326_asn97_A"][f_idx])
        zn_val = float(metrics["zn_dists_A"][f_idx].mean())

        draw.rounded_rectangle([18, 16, 520, 86], radius=10, fill=(248, 250, 252), outline=(203, 213, 225), width=2)
        draw.text((32, 24), f"{label}   |   {t_ns:5.1f} / {total_ns:.1f} ns", fill=(15, 23, 42), font=font_title)
        draw.text((32, 56), "ff14SB + OPC + ZAFF C4 Zn²⁺  (4 fs HMR, 300 K)", fill=(71, 85, 105), font=font_sub)

        bar_x0, bar_x1, bar_y0, bar_y1 = 18, 520, 92, 100
        draw.rounded_rectangle([bar_x0, bar_y0, bar_x1, bar_y1], radius=4, fill=(226, 232, 240))
        prog_x = bar_x0 + int((bar_x1 - bar_x0) * min(max(t_ns / max(total_ns, 1e-6), 0.0), 1.0))
        if prog_x > bar_x0 + 4:
            draw.rounded_rectangle([bar_x0, bar_y0, prog_x, bar_y1], radius=4, fill=(37, 99, 235))

        draw.rounded_rectangle([18, lh - 52, lw - 18, lh - 14], radius=8, fill=(248, 250, 252), outline=(203, 213, 225), width=1)
        lx = 32
        for col, lbl in legend_items:
            draw.rounded_rectangle([lx, lh - 40, lx + 14, lh - 26], radius=3, fill=col)
            draw.text((lx + 20, lh - 41), lbl, fill=(30, 41, 59), font=font_legend)
            lx += 148 if len(lbl) > 6 else 118

        draw.line([(lw, 0), (lw, lh)], fill=(203, 213, 225), width=2)

        vals = [rmsd_val, r326_val, zn_val]
        dot_colors = [(29, 78, 216), (220, 38, 38), (109, 40, 217)]
        labels = [f"{rmsd_val:.2f} Å", f"{r326_val:.2f} Å", f"{zn_val:.2f} Å"]
        frac_x = min(max((t_ns - plot_meta["xmin"]) / max(plot_meta["xmax"] - plot_meta["xmin"], 1e-6), 0.0), 1.0)

        for bb, val, dcol, vtxt in zip(plot_meta["bboxes"], vals, dot_colors, labels):
            px = lw + bb["x0"] + frac_x * (bb["x1"] - bb["x0"])
            frac_y = min(max((val - bb["ymin"]) / max(bb["ymax"] - bb["ymin"], 1e-6), 0.0), 1.0)
            py = bb["y_bot"] - frac_y * (bb["y_bot"] - bb["y_top"])
            draw.line([(px, bb["y_top"]), (px, bb["y_bot"])], fill=(239, 68, 68), width=2)
            r = 5
            draw.ellipse([px - r, py - r, px + r, py + r], fill=dcol, outline=(255, 255, 255), width=2)
            tx = lw + bb["x0"] + 6
            ty = bb["y_top"] + 5
            draw.rounded_rectangle([tx, ty, tx + 68, ty + 20], radius=4, fill=(255, 255, 255), outline=dcol, width=1)
            draw.text((tx + 6, ty + 2), vtxt, fill=dcol, font=font_readout)

        canvas.save(comp_dir / f"frame_{st:04d}.png")
        if st == 1 or st % 40 == 0 or st == n_states:
            print(f"    [HUD] Composited frame {st}/{n_states}", flush=True)


def encode_videos(comp_dir: Path, out_mp4: Path, out_gif: Path, fps: int = 15) -> None:
    print(f"[STEP 4] Encoding H.264 MP4 -> {out_mp4} ({fps} fps) ...", flush=True)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(comp_dir / "frame_%04d.png"),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(out_mp4),
        ],
        check=True,
    )
    print(f"[STEP 4] Encoding compact GIF preview -> {out_gif} ...", flush=True)
    palette = comp_dir / "palette.png"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(comp_dir / "frame_%04d.png"),
            "-vf", "fps=12,scale=740:-1:flags=lanczos,palettegen",
            str(palette),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", str(comp_dir / "frame_%04d.png"),
            "-i", str(palette),
            "-lavfi", "fps=12,scale=740:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer",
            str(out_gif),
        ],
        check=True,
    )
    print(f"[DONE] MP4 ({out_mp4.stat().st_size / 1e6:.2f} MB) & GIF ({out_gif.stat().st_size / 1e6:.2f} MB)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render CPU-only 4XXB trajectory animation with PyMOL + live plots.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--replica", type=int, default=1)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--label", default=None, help="HUD & plot title label (default: '<run_dir.name> rep<N>').")
    ap.add_argument("--save-ps", type=float, default=20.0, help="Frame save stride in ps (default: 20.0).")
    ap.add_argument("--n-frames", type=int, default=201)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--preview-only", action="store_true", help="Render only frame 1 to inspect camera angle.")
    ap.add_argument("--skip-prep", action="store_true", help="Reuse existing traj_movie.pdb and movie_metrics.npz.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    frames_dir = outdir / "raw_frames"
    comp_dir = outdir / "comp_frames"
    label = args.label or f"{run_dir.name} rep{args.replica}"

    if args.skip_prep and (outdir / "movie_metrics.npz").exists() and (outdir / "traj_movie.pdb").exists():
        npz = np.load(outdir / "movie_metrics.npz")
        metrics = {k: npz[k] for k in npz.files}
        metrics["movie_pdb"] = str(outdir / "traj_movie.pdb")
    else:
        metrics = prepare_4xxb_data(run_dir, args.replica, outdir, args.n_frames, save_ps=args.save_ps)

    total_tag = f"{int(round(float(metrics['times_ns'][-1])))}ns"
    pse_path = outdir / f"{run_dir.name}_rep{args.replica}_{total_tag}.pse"
    render_pymol_frames(
        Path(metrics["movie_pdb"]),
        frames_dir,
        pse_path,
        n_states=len(metrics["frame_indices"]),
        max_threads=args.threads,
        preview_only=args.preview_only,
    )
    composite_frames(metrics, frames_dir, comp_dir, label=label, preview_only=args.preview_only)

    if not args.preview_only:
        out_mp4 = outdir / f"{run_dir.name}_rep{args.replica}_{total_tag}.mp4"
        out_gif = outdir / f"{run_dir.name}_rep{args.replica}_{total_tag}.gif"
        encode_videos(comp_dir, out_mp4, out_gif, fps=args.fps)


if __name__ == "__main__":
    main()
