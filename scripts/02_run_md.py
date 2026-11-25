#!/usr/bin/env python3
"""
02_run_md.py
Production OpenMM 8 MD runner implementing:
  1. Roe & Brooks (2020) 10-step relaxation protocol:
     - Double-precision CPU energy minimisations (Steps 1, 3, 4, 5) to prevent GPU fixed-precision overflow
     - Periodic-safe harmonic Cartesian restraints (5.0 -> 2.0 -> 0.1 -> 0 kcal/(mol*A^2) EM;
       5.0 NVT -> 1.0 NPT -> 0.5 heavy NPT -> 0.5 backbone NPT -> 0 unrestrained NPT)
     - 1 ns unrestrained density-stabilisation stage (Step 10) at 2 fs prior to 4 fs HMR production
  2. Production NPT at 300 K, 1 bar using LangevinMiddleIntegrator (dt = 4 fs with HMR prmtop;
     Hopkins et al. 2015) + MonteCarloBarostat + PME (1.0 nm cutoff, analytical LJ dispersion correction)
  3. Cross-GPU portable checkpointing via XML state files (`state.xml`) every `--chunk-ns` (default 5 ns)
     so interrupted replicas resume cleanly across heterogeneous GPUs without binary `.chk` errors.
  4. Streaming solvent/ion stripping upon completion -> lightweight `prod_dry.dcd` for CPU post-processing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import List

import mdtraj as md
from openmm import (
    CustomExternalForce,
    LangevinMiddleIntegrator,
    MonteCarloBarostat,
    Platform,
    XmlSerializer,
    unit,
)
from openmm.app import (
    AmberInpcrdFile,
    AmberPrmtopFile,
    DCDReporter,
    HBonds,
    PME,
    Simulation,
    StateDataReporter,
)

KCAL_MOL_A2_TO_KJ_MOL_NM2 = 418.4  # 1 kcal/(mol*A^2) = 4.184 * 100 kJ/(mol*nm^2)
BACKBONE_ATOMS = {"N", "CA", "C", "O", "P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'"}
SOLVENT_RESNAMES = {"WAT", "HOH", "TIP3", "OPC", "K+", "Cl-", "Na+", "CL", "NA", "K", "MG"}


def get_platform(platform_name: str, for_minimization: bool = False) -> tuple[Platform, dict]:
    if for_minimization or platform_name.upper() == "CPU":
        return Platform.getPlatformByName("CPU"), {}
    plat = Platform.getPlatformByName(platform_name)
    props = {"Precision": "mixed"} if platform_name.upper() in ("CUDA", "OPENCL") else {}
    return plat, props


def collect_restraint_indices(topology) -> tuple[List[int], List[int]]:
    heavy_solute: List[int] = []
    backbone_solute: List[int] = []
    for atom in topology.atoms():
        if atom.residue.name in SOLVENT_RESNAMES:
            continue
        if atom.element is None or atom.element.symbol == "H":
            continue
        heavy_solute.append(atom.index)
        if atom.name in BACKBONE_ATOMS or atom.residue.name in ("ZN1", "ZN2"):
            backbone_solute.append(atom.index)
    return heavy_solute, backbone_solute


def build_system(
    prmtop: AmberPrmtopFile,
    use_constraints: bool,
    add_barostat: bool,
    temperature_k: float = 300.0,
    pressure_bar: float = 1.0,
    restraint_indices: List[int] | None = None,
    restraint_k_kcal_a2: float = 0.0,
    ref_positions=None,
):
    system = prmtop.createSystem(
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=HBonds if use_constraints else None,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
    )
    if add_barostat:
        system.addForce(
            MonteCarloBarostat(pressure_bar * unit.bar, temperature_k * unit.kelvin, 25)
        )
    if restraint_indices and restraint_k_kcal_a2 > 0.0 and ref_positions is not None:
        k_val = restraint_k_kcal_a2 * KCAL_MOL_A2_TO_KJ_MOL_NM2
        restraint = CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
        restraint.addGlobalParameter("k", k_val * unit.kilojoules_per_mole / unit.nanometer**2)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        for idx in restraint_indices:
            pos = ref_positions[idx].value_in_unit(unit.nanometer)
            restraint.addParticle(idx, [pos[0], pos[1], pos[2]])
        system.addForce(restraint)
    return system


def run_minimization_cpu(
    prmtop: AmberPrmtopFile,
    positions,
    box_vectors,
    restraint_indices: List[int],
    k_kcal_a2: float,
    ref_positions,
    max_iter: int = 1000,
):
    """Double-precision CPU energy minimisation (Roe & Brooks 2020 safeguard against GPU overflow)."""
    system = build_system(
        prmtop,
        use_constraints=False,
        add_barostat=False,
        restraint_indices=restraint_indices,
        restraint_k_kcal_a2=k_kcal_a2,
        ref_positions=ref_positions,
    )
    integrator = LangevinMiddleIntegrator(300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picosecond)
    plat, props = get_platform("CPU", for_minimization=True)
    sim = Simulation(prmtop.topology, system, integrator, plat, props)
    sim.context.setPeriodicBoxVectors(*box_vectors)
    sim.context.setPositions(positions)
    e0 = sim.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    sim.minimizeEnergy(maxIterations=max_iter)
    st = sim.context.getState(getPositions=True, getEnergy=True)
    e1 = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    print(f"    [EM CPU] k={k_kcal_a2:.1f} kcal/(mol*A^2) | E: {e0:,.1f} -> {e1:,.1f} kJ/mol", flush=True)
    return st.getPositions(), st.getPeriodicBoxVectors()


def run_md_stage(
    prmtop: AmberPrmtopFile,
    positions,
    velocities,
    box_vectors,
    n_steps: int,
    dt_ps: float,
    add_barostat: bool,
    restraint_indices: List[int],
    k_kcal_a2: float,
    ref_positions,
    platform_name: str,
    seed: int,
    temperature_k: float = 300.0,
    friction_ps: float = 5.0,
    log_csv: Path | None = None,
    label: str = "",
):
    system = build_system(
        prmtop,
        use_constraints=True,
        add_barostat=add_barostat,
        temperature_k=temperature_k,
        restraint_indices=restraint_indices,
        restraint_k_kcal_a2=k_kcal_a2,
        ref_positions=ref_positions,
    )
    integrator = LangevinMiddleIntegrator(
        temperature_k * unit.kelvin, friction_ps / unit.picosecond, dt_ps * unit.picoseconds
    )
    integrator.setRandomNumberSeed(seed)
    plat, props = get_platform(platform_name, for_minimization=False)
    sim = Simulation(prmtop.topology, system, integrator, plat, props)
    sim.context.setPeriodicBoxVectors(*box_vectors)
    sim.context.setPositions(positions)
    if velocities is None:
        sim.context.setVelocitiesToTemperature(temperature_k * unit.kelvin, seed)
    else:
        sim.context.setVelocities(velocities)
    if log_csv is not None:
        sim.reporters.append(
            StateDataReporter(
                str(log_csv),
                max(500, n_steps // 100),
                step=True,
                time=True,
                potentialEnergy=True,
                temperature=True,
                density=True,
                speed=True,
            )
        )
    sim.step(n_steps)
    st = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    pe = st.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
    bv = st.getPeriodicBoxVectors()
    print(f"    [{label}] {n_steps * dt_ps:.1f} ps done | PE={pe:,.1f} kJ/mol", flush=True)
    return st.getPositions(), st.getVelocities(), bv, st


def run_roe_brooks_10step_relaxation(
    prmtop_std: AmberPrmtopFile,
    inpcrd: AmberInpcrdFile,
    rep_dir: Path,
    platform_name: str,
    seed: int,
    temperature_k: float = 300.0,
    step10_ns: float = 1.0,
) -> Path:
    equil_xml = rep_dir / "equil_final.xml"
    if equil_xml.exists():
        print(f"[RESUME] Equilibration already complete ({equil_xml}). Skipping 10-step relaxation.", flush=True)
        return equil_xml

    heavy_idx, bb_idx = collect_restraint_indices(prmtop_std.topology)
    init_pos = inpcrd.positions
    box_vec = inpcrd.boxVectors
    print(f"[RELAX] Starting Roe & Brooks 10-step relaxation ({len(heavy_idx)} heavy solute atoms, {len(bb_idx)} backbone)", flush=True)

    # Step 1: EM mobile solvent/ions, heavy solute k=5.0
    pos, box_vec = run_minimization_cpu(prmtop_std, init_pos, box_vec, heavy_idx, 5.0, init_pos, 1000)
    # Step 2: 15 ps NVT (1 fs), heavy solute k=5.0
    pos, vel, box_vec, _ = run_md_stage(
        prmtop_std, pos, None, box_vec, 15_000, 0.001, False, heavy_idx, 5.0, init_pos,
        platform_name, seed + 2, temperature_k, friction_ps=5.0, label="Step 2 NVT k=5.0"
    )
    # Steps 3-5: progressive CPU EM (2.0 -> 0.1 -> 0.0)
    pos, box_vec = run_minimization_cpu(prmtop_std, pos, box_vec, heavy_idx, 2.0, init_pos, 1000)
    pos, box_vec = run_minimization_cpu(prmtop_std, pos, box_vec, heavy_idx, 0.1, init_pos, 1000)
    pos, box_vec = run_minimization_cpu(prmtop_std, pos, box_vec, [], 0.0, pos, 1000)
    ref_step5 = pos

    # Step 6: 5 ps NPT (1 fs), heavy k=1.0, fresh velocities
    pos, vel, box_vec, _ = run_md_stage(
        prmtop_std, pos, None, box_vec, 5_000, 0.001, True, heavy_idx, 1.0, ref_step5,
        platform_name, seed + 6, temperature_k, friction_ps=5.0, label="Step 6 NPT k=1.0"
    )
    # Step 7: 5 ps NPT (1 fs), heavy k=0.5
    pos, vel, box_vec, _ = run_md_stage(
        prmtop_std, pos, vel, box_vec, 5_000, 0.001, True, heavy_idx, 0.5, ref_step5,
        platform_name, seed + 7, temperature_k, friction_ps=5.0, label="Step 7 NPT k=0.5"
    )
    # Step 8: 10 ps NPT (1 fs), backbone k=0.5
    pos, vel, box_vec, _ = run_md_stage(
        prmtop_std, pos, vel, box_vec, 10_000, 0.001, True, bb_idx, 0.5, ref_step5,
        platform_name, seed + 8, temperature_k, friction_ps=5.0, label="Step 8 NPT bb k=0.5"
    )
    # Step 9: 10 ps NPT (2 fs), unrestrained
    pos, vel, box_vec, _ = run_md_stage(
        prmtop_std, pos, vel, box_vec, 5_000, 0.002, True, [], 0.0, None,
        platform_name, seed + 9, temperature_k, friction_ps=2.0, label="Step 9 NPT unrestrained"
    )
    # Step 10: 1 ns NPT density stabilisation (2 fs, production gamma=1.0 ps^-1)
    n_step10 = int(round(step10_ns * 1000.0 / 0.002))
    pos, vel, box_vec, st_final = run_md_stage(
        prmtop_std, pos, vel, box_vec, n_step10, 0.002, True, [], 0.0, None,
        platform_name, seed + 10, temperature_k, friction_ps=1.0,
        log_csv=rep_dir / "equil_step10.csv", label=f"Step 10 NPT density ({step10_ns} ns)"
    )

    equil_xml.write_text(XmlSerializer.serialize(st_final), encoding="utf-8")
    print(f"[OK] Saved equilibrated XML state -> {equil_xml}", flush=True)
    return equil_xml


def write_status(status_path: Path, data: dict) -> None:
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(status_path)


def strip_solvent_to_dry_dcd(build_dir: Path, rep_dir: Path) -> None:
    solv_prmtop = build_dir / "solv.prmtop"
    prod_dcd = rep_dir / "prod.dcd"
    dry_dcd = rep_dir / "prod_dry.dcd"
    if not prod_dcd.exists():
        return
    print(f"[POST] Stripping solvent/ions -> {dry_dcd.name} (streaming chunks for low RAM)", flush=True)
    top = md.load_prmtop(str(solv_prmtop))
    solute_idx = [a.index for a in top.atoms if a.residue.name not in SOLVENT_RESNAMES]
    chunks = []
    for chunk in md.iterload(str(prod_dcd), top=top, chunk=500, atom_indices=solute_idx):
        chunks.append(chunk)
    if chunks:
        full_dry = md.join(chunks)
        full_dry.save_dcd(str(dry_dcd))
        print(f"[OK] Wrote {dry_dcd.name}: {full_dry.n_frames} frames, {full_dry.n_atoms} solute atoms", flush=True)


def run_production(
    build_dir: Path,
    rep_dir: Path,
    equil_xml: Path,
    target_ns: float,
    chunk_ns: float,
    save_interval_ps: float,
    dt_fs: float,
    platform_name: str,
    seed: int,
    temperature_k: float = 300.0,
) -> None:
    prmtop_file = build_dir / ("solv_hmr.prmtop" if dt_fs >= 3.0 else "solv.prmtop")
    prmtop = AmberPrmtopFile(str(prmtop_file))
    state_xml = rep_dir / "state.xml"
    status_json = rep_dir / "status.json"
    prod_dcd = rep_dir / "prod.dcd"
    prod_csv = rep_dir / "prod_energy.csv"

    completed_ns = 0.0
    if status_json.exists():
        st_data = json.loads(status_json.read_text(encoding="utf-8"))
        completed_ns = float(st_data.get("completed_ns", 0.0))

    if completed_ns >= target_ns - 1e-6:
        print(f"[DONE] Replica already reached {completed_ns:.2f}/{target_ns:.2f} ns.", flush=True)
        if not (rep_dir / "prod_dry.dcd").exists():
            strip_solvent_to_dry_dcd(build_dir, rep_dir)
        return

    dt_ps = dt_fs / 1000.0
    report_steps = max(1, int(round(save_interval_ps / dt_ps)))

    system = build_system(
        prmtop,
        use_constraints=True,
        add_barostat=True,
        temperature_k=temperature_k,
        pressure_bar=1.0,
    )
    integrator = LangevinMiddleIntegrator(
        temperature_k * unit.kelvin, 1.0 / unit.picosecond, dt_ps * unit.picoseconds
    )
    integrator.setRandomNumberSeed(seed + int(completed_ns * 100))
    plat, props = get_platform(platform_name, for_minimization=False)
    sim = Simulation(prmtop.topology, system, integrator, plat, props)

    is_resume = state_xml.exists() and completed_ns > 0.0 and prod_dcd.exists()
    load_path = state_xml if is_resume else equil_xml
    print(f"[PROD] Loading portable XML state from {load_path.name} (resume={is_resume}, completed={completed_ns:.2f} ns)", flush=True)
    sim.context.setState(XmlSerializer.deserialize(load_path.read_text(encoding="utf-8")))
    if not is_resume:
        sim.context.setTime(0.0 * unit.picosecond)
        sim.currentStep = 0

    sim.reporters.append(DCDReporter(str(prod_dcd), report_steps, append=is_resume))
    sim.reporters.append(
        StateDataReporter(
            str(prod_csv),
            report_steps,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            append=is_resume,
        )
    )
    sim.reporters.append(
        StateDataReporter(
            sys.stdout,
            report_steps * 5,
            step=True,
            time=True,
            potentialEnergy=True,
            temperature=True,
            density=True,
            speed=True,
        )
    )

    while completed_ns < target_ns - 1e-6:
        this_chunk = min(chunk_ns, target_ns - completed_ns)
        n_steps = int(round((this_chunk * 1000.0) / dt_ps))
        t0 = time.time()
        sim.step(n_steps)
        elapsed_min = (time.time() - t0) / 60.0

        completed_ns = round(completed_ns + this_chunk, 4)
        cur_state = sim.context.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=False)
        tmp_xml = state_xml.with_suffix(".xml.tmp")
        tmp_xml.write_text(XmlSerializer.serialize(cur_state), encoding="utf-8")
        tmp_xml.replace(state_xml)

        write_status(
            status_json,
            {
                "completed_ns": completed_ns,
                "target_ns": target_ns,
                "dt_fs": dt_fs,
                "platform": platform_name,
                "seed": seed,
                "last_chunk_min": round(elapsed_min, 2),
                "est_ns_per_day": round(this_chunk / (elapsed_min / 1440.0), 1) if elapsed_min > 0 else 0.0,
                "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        print(
            f"[CHECKPOINT] {completed_ns:.2f}/{target_ns:.2f} ns saved to state.xml "
            f"({elapsed_min:.1f} min, ~{this_chunk / max(elapsed_min / 1440.0, 1e-6):.1f} ns/day)",
            flush=True,
        )

    strip_solvent_to_dry_dcd(build_dir, rep_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Roe & Brooks 10-step relaxation + 4 fs HMR OpenMM production MD.")
    ap.add_argument("--run-dir", required=True, help="System/variant root directory containing build/ subfolder.")
    ap.add_argument("--replica", type=int, required=True, help="Replica index (1, 2, 3...).")
    ap.add_argument("--ns", type=float, default=200.0, help="Target production length in ns (default: 200).")
    ap.add_argument("--chunk-ns", type=float, default=5.0, help="Portable state.xml checkpoint chunk in ns (default: 5).")
    ap.add_argument("--save-ps", type=float, default=20.0, help="Frame save stride in ps (default: 20 ps -> 10,000 frames per 200 ns).")
    ap.add_argument("--dt-fs", type=float, default=4.0, help="Production timestep in fs (default: 4.0 with HMR; set 2.0 for non-HMR).")
    ap.add_argument("--step10-ns", type=float, default=1.0, help="Density-stabilisation Step 10 length in ns (default: 1.0).")
    ap.add_argument("--platform", default="CUDA", choices=["CUDA", "OpenCL", "CPU"], help="OpenMM GPU/CPU platform.")
    ap.add_argument("--temp", type=float, default=300.0, help="Simulation temperature in K (default: 300).")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    build_dir = run_dir / "build"
    rep_dir = run_dir / f"rep{args.replica}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    seed = 1000 * args.replica + 42
    prmtop_std = AmberPrmtopFile(str(build_dir / "solv.prmtop"))
    inpcrd = AmberInpcrdFile(str(build_dir / "solv.inpcrd"))

    equil_xml = run_roe_brooks_10step_relaxation(
        prmtop_std, inpcrd, rep_dir, args.platform, seed, temperature_k=args.temp, step10_ns=args.step10_ns
    )
    run_production(
        build_dir=build_dir,
        rep_dir=rep_dir,
        equil_xml=equil_xml,
        target_ns=args.ns,
        chunk_ns=args.chunk_ns,
        save_interval_ps=args.save_ps,
        dt_fs=args.dt_fs,
        platform_name=args.platform,
        seed=seed,
        temperature_k=args.temp,
    )


if __name__ == "__main__":
    main()
