#!/usr/bin/env python3
"""
01_build_system.py
Modular, reproducible system builder for:
  - 4XXB       : Human MDM2(293-334) + RPL11(13-178, internal loop 139-153 rebuilt) + ZAFF C4 Zn2+ (Center ID 1)
  - 2AC0_dimer : Human p53 DBD dimer (Chains A+B, 94-291) + 12-bp DNA (Chains E+F) + 2x ZAFF CCCH Zn2+ (Center ID 2)

Force field stack (validated for ZAFF + DNA + OPC compatibility):
  - Protein : AMBER ff14SB
  - DNA     : AMBER DNA.OL15
  - Water   : OPC (4-point) + Li-Merz 12-6 OPC monovalent ion parameters (0.15 M KCl)
  - Zinc    : AMBER ZAFF bonded model (Center ID 1 for MDM2 CCCC; Center ID 2 for p53 CCCH-HIE[ND1])
  - HMR     : ParmEd HMassRepartition (3.024 Da solute H; water untouched) for 4 fs production MD

Example mutations:
  --mutations "A:R280K,B:R280K"   (homozygous p53 dimer mutant)
  --mutations "A:R280T"           (heterozygous WT/Mut p53 dimer)
  --mutations "B:R326H"           (MDM2 R326H / COSMIC R332H in 4XXB)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

AA1_TO_3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

SYSTEM_CONFIGS = {
    "4XXB": {
        "pdb_id": "4XXB",
        "keep_chains": ["A", "B"],
        "protein_chains": ["A", "B"],
        "dna_chains": [],
        "trim_ranges": {"A": (13, 178), "B": (293, 334)},
        "cap_chains": ["A", "B"],
        "box_pad_angstrom": 12.0,
        "zaff_centers": [
            {
                "center_id": 1,
                "zn_orig": ("B", 501),
                "zn_resname": "ZN1",
                "ligands": [
                    ("B", 305, "CY1", "SG"),
                    ("B", 308, "CY1", "SG"),
                    ("B", 319, "CY1", "SG"),
                    ("B", 322, "CY1", "SG"),
                ],
                "atom_types": [("ZN", "Zn", "sp3"), ("S1", "S", "sp3")],
            }
        ],
    },
    "2AC0_dimer": {
        "pdb_id": "2AC0",
        "keep_chains": ["A", "B", "E", "F"],
        "protein_chains": ["A", "B"],
        "dna_chains": ["E", "F"],
        # Trim Chain A to 94-291 to match Chain B (94-291) symmetrically
        "trim_ranges": {"A": (94, 291), "B": (94, 291), "E": (1, 12), "F": (1, 12)},
        "cap_chains": ["A", "B"],
        "box_pad_angstrom": 12.0,
        "zaff_centers": [
            {
                "center_id": 2,
                "zn_orig": ("A", 1),
                "zn_resname": "ZN2",
                "ligands": [
                    ("A", 176, "CY2", "SG"),
                    ("A", 179, "HE1", "ND1"),
                    ("A", 238, "CY2", "SG"),
                    ("A", 242, "CY2", "SG"),
                ],
                "atom_types": [("ZN", "Zn", "sp3"), ("S2", "S", "sp3"), ("N1", "N", "sp3")],
            },
            {
                "center_id": 2,
                "zn_orig": ("B", 1),
                "zn_resname": "ZN2",
                "ligands": [
                    ("B", 176, "CY2", "SG"),
                    ("B", 179, "HE1", "ND1"),
                    ("B", 238, "CY2", "SG"),
                    ("B", 242, "CY2", "SG"),
                ],
                "atom_types": [("ZN", "Zn", "sp3"), ("S2", "S", "sp3"), ("N1", "N", "sp3")],
            },
        ],
    },
}


def download_pdb(pdb_id: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"[INFO] Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    return dest


def parse_mutations(mut_str: str) -> Dict[str, List[str]]:
    """Parse 'A:R280K,B:R280K' into {'A': ['ARG-280-LYS'], 'B': ['ARG-280-LYS']}."""
    out: Dict[str, List[str]] = {}
    if not mut_str or mut_str.strip().upper() in ("WT", "NONE", ""):
        return out
    for token in mut_str.split(","):
        token = token.strip()
        if not token:
            continue
        m = re.fullmatch(r"([A-Za-z0-9]):([A-Za-z])(\d+)([A-Za-z])", token)
        if not m:
            raise ValueError(
                f"Invalid mutation token '{token}'. Expected format 'CHAIN:X123Y', e.g. 'A:R280K' or 'B:R326H'."
            )
        ch, wt1, pos, mut1 = m.group(1), m.group(2).upper(), m.group(3), m.group(4).upper()
        if wt1 not in AA1_TO_3 or mut1 not in AA1_TO_3:
            raise ValueError(f"Unsupported amino acid code in '{token}'.")
        out.setdefault(ch, []).append(f"{AA1_TO_3[wt1]}-{pos}-{AA1_TO_3[mut1]}")
    return out


def extract_zn_coords(raw_pdb: Path, zn_specs: List[Tuple[str, int]]) -> Dict[Tuple[str, int], np.ndarray]:
    coords: Dict[Tuple[str, int], np.ndarray] = {}
    wanted = set(zn_specs)
    for line in raw_pdb.read_text(encoding="utf-8").splitlines():
        if line.startswith(("HETATM", "ATOM  ")) and line[17:20].strip() == "ZN":
            ch = line[21].strip()
            ri = int(line[22:26])
            if (ch, ri) in wanted:
                coords[(ch, ri)] = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float
                )
    missing = wanted - set(coords.keys())
    if missing:
        raise RuntimeError(f"Could not locate required ZN ions {missing} in {raw_pdb}")
    return coords


def clean_and_mutate_with_pdbfixer(
    raw_pdb: Path,
    cfg: dict,
    mut_dict: Dict[str, List[str]],
    fixed_pdb: Path,
) -> None:
    from openmm.app import PDBFile
    from pdbfixer import PDBFixer

    fixer = PDBFixer(filename=str(raw_pdb))

    all_chains = [c.id for c in fixer.topology.chains()]
    remove_idx = [i for i, cid in enumerate(all_chains) if cid not in cfg["keep_chains"]]
    if remove_idx:
        fixer.removeChains(remove_idx)

    fixer.removeHeterogens(keepWater=False)

    # Rebuild internal missing loops only; discard terminal overhangs (critical for 4XXB MDM2 335-437)
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    for key in list(fixer.missingResidues.keys()):
        chain_idx, res_idx = key
        chain_len = len(list(chains[chain_idx].residues()))
        if res_idx == 0 or res_idx >= chain_len:
            del fixer.missingResidues[key]

    if fixer.missingResidues:
        print(f"[INFO] Rebuilding internal missing loops only: {fixer.missingResidues}")

    for ch_id, muts in mut_dict.items():
        print(f"[INFO] Applying mutations on chain {ch_id}: {muts}")
        fixer.applyMutations(muts, ch_id)

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    with fixed_pdb.open("w", encoding="utf-8") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)


def compute_cap_coords(
    n_coord: np.ndarray, ca_coord: np.ndarray, c_coord: np.ndarray, is_nterm: bool
) -> np.ndarray:
    """Place ACE C (1.33 A from N) or NME N (1.33 A from C) along idealised peptide vector."""
    if is_nterm:
        v = n_coord - ca_coord
        v = v / np.linalg.norm(v)
        return n_coord + 1.33 * v
    v = c_coord - ca_coord
    v = v / np.linalg.norm(v)
    return c_coord + 1.33 * v


def format_pdb_line(
    record: str, serial: int, atom_name: str, resname: str, chain: str, resseq: int, xyz: np.ndarray, element: str
) -> str:
    an = f" {atom_name:<3s}" if len(atom_name) < 4 else f"{atom_name:<4s}"
    return (
        f"{record:<6s}{serial:5d} {an:4s} {resname:>3s} {chain:1s}{resseq:4d}    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}  1.00  0.00          {element:>2s}"
    )


def prepare_zaff_capped_pdb(
    fixed_pdb: Path,
    cfg: dict,
    zn_coords: Dict[Tuple[str, int], np.ndarray],
    out_pdb: Path,
    map_json: Path,
) -> Tuple[List[Tuple[int, int, str]], dict]:
    """
    Strip hydrogens, trim terminal overhangs symmetrically, add ACE/NME anchor atoms for protein chains,
    rename ZAFF coordinating residues (CY1/CY2/HE1), append ZN1/ZN2 separated by TER cards, and record
    exact 1-based tleap residue indices for bond commands and downstream analysis.
    """
    chain_order: List[str] = []
    residues: Dict[Tuple[str, int], Tuple[str, List[Tuple[str, np.ndarray, str]]]] = {}

    for line in fixed_pdb.read_text(encoding="utf-8").splitlines():
        if not line.startswith("ATOM  "):
            continue
        an = line[12:16].strip()
        elem = line[76:78].strip() if len(line) >= 78 else ""
        if elem == "H" or an.startswith("H") or (len(an) >= 2 and an[0].isdigit() and an[1] == "H"):
            continue
        ch = line[21].strip()
        if ch not in cfg["keep_chains"]:
            continue
        ri = int(line[22:26])
        lo, hi = cfg["trim_ranges"][ch]
        if ri < lo or ri > hi:
            continue
        rn = line[17:20].strip()
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
        if not elem:
            elem = re.sub(r"[^A-Za-z]", "", an)[:1]
        if ch not in chain_order:
            chain_order.append(ch)
        key = (ch, ri)
        if key not in residues:
            residues[key] = (rn, [])
        residues[key][1].append((an, xyz, elem))

    zaff_rename: Dict[Tuple[str, int], str] = {}
    for zc in cfg["zaff_centers"]:
        for lch, lri, lren, _ in zc["ligands"]:
            zaff_rename[(lch, lri)] = lren

    pdb_lines: List[str] = []
    res_map: List[dict] = []
    orig_to_tleap: Dict[Tuple[str, int], int] = {}
    bond_specs: List[Tuple[int, int, str]] = []
    placed_zn: set[Tuple[str, int]] = set()
    serial = 1
    tleap_idx = 0

    def append_zn_center(zc: dict) -> None:
        nonlocal serial, tleap_idx
        zch, zri = zc["zn_orig"]
        xyz = zn_coords[(zch, zri)]
        tleap_idx += 1
        zn_tleap_idx = tleap_idx
        pdb_lines.append(format_pdb_line("HETATM", serial, "ZN", zc["zn_resname"], "Z", zn_tleap_idx, xyz, "ZN"))
        serial += 1
        pdb_lines.append("TER")
        res_map.append({"tleap_idx": zn_tleap_idx, "chain": zch, "pdb_resseq": zri, "resname": zc["zn_resname"], "role": "zn"})
        for lch, lri, _, latom in zc["ligands"]:
            lig_tleap_idx = orig_to_tleap[(lch, lri)]
            bond_specs.append((zn_tleap_idx, lig_tleap_idx, latom))
        placed_zn.add((zch, zri))

    for ch in cfg["keep_chains"]:
        ch_keys = sorted([k for k in residues if k[0] == ch], key=lambda x: x[1])
        if not ch_keys:
            raise RuntimeError(f"Chain {ch} has no atoms after filtering!")

        if ch in cfg["cap_chains"]:
            first_atoms = {an: xyz for an, xyz, _ in residues[ch_keys[0]][1]}
            ace_c = compute_cap_coords(first_atoms["N"], first_atoms["CA"], first_atoms["C"], is_nterm=True)
            tleap_idx += 1
            pdb_lines.append(format_pdb_line("ATOM", serial, "C", "ACE", ch, tleap_idx, ace_c, "C"))
            serial += 1
            res_map.append({"tleap_idx": tleap_idx, "chain": ch, "pdb_resseq": ch_keys[0][1] - 1, "resname": "ACE", "role": "cap"})

        for _, ri in ch_keys:
            rn, atom_list = residues[(ch, ri)]
            out_rn = zaff_rename.get((ch, ri), rn)
            tleap_idx += 1
            orig_to_tleap[(ch, ri)] = tleap_idx
            role = "dna" if ch in cfg["dna_chains"] else "protein"
            res_map.append({"tleap_idx": tleap_idx, "chain": ch, "pdb_resseq": ri, "resname": out_rn, "role": role})
            for an, xyz, elem in atom_list:
                if role == "dna" and ri == ch_keys[0][1] and an in ("P", "OP1", "OP2", "O1P", "O2P", "OP3"):
                    continue
                if ch in cfg["cap_chains"] and an == "OXT":
                    continue
                pdb_lines.append(format_pdb_line("ATOM", serial, an, out_rn, ch, tleap_idx, xyz, elem))
                serial += 1

        if ch in cfg["cap_chains"]:
            last_atoms = {an: xyz for an, xyz, _ in residues[ch_keys[-1]][1]}
            nme_n = compute_cap_coords(last_atoms["N"], last_atoms["CA"], last_atoms["C"], is_nterm=False)
            tleap_idx += 1
            pdb_lines.append(format_pdb_line("ATOM", serial, "N", "NME", ch, tleap_idx, nme_n, "N"))
            serial += 1
            res_map.append({"tleap_idx": tleap_idx, "chain": ch, "pdb_resseq": ch_keys[-1][1] + 1, "resname": "NME", "role": "cap"})

        pdb_lines.append("TER")

        # Place each ZAFF ZN immediately after its coordinating chain so every bonded molecule
        # is contiguous in atom order (prevents ParmEd HMR from reordering atoms relative to solv.prmtop)
        for zc in cfg["zaff_centers"]:
            if zc["zn_orig"] not in placed_zn and all((lch, lri) in orig_to_tleap for lch, lri, _, _ in zc["ligands"]):
                append_zn_center(zc)

    for zc in cfg["zaff_centers"]:
        if zc["zn_orig"] not in placed_zn:
            append_zn_center(zc)

    pdb_lines.append("END")
    out_pdb.write_text("\n".join(pdb_lines) + "\n", encoding="utf-8")

    meta = {
        "system": cfg["pdb_id"],
        "n_solute_residues": tleap_idx,
        "bonds": [{"zn_idx": z, "lig_idx": l, "lig_atom": a} for z, l, a in bond_specs],
        "residues": res_map,
    }
    map_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return bond_specs, meta


def write_and_run_tleap(
    prep_pdb: Path,
    bond_specs: List[Tuple[int, int, str]],
    cfg: dict,
    params_dir: Path,
    work_dir: Path,
    salt_molarity: float = 0.15,
) -> Tuple[Path, Path]:
    import parmed as pmd

    zaff_prep = params_dir / "ZAFF_H.prep"
    zaff_frcmod = params_dir / "ZAFF.frcmod"
    if not zaff_prep.exists() or not zaff_frcmod.exists():
        raise FileNotFoundError(f"Missing ZAFF files in {params_dir}. Expected ZAFF_H.prep and ZAFF.frcmod.")

    # Use relative paths inside generated LEaP scripts so artefacts remain portable across hosts
    rel_prep = os.path.relpath(zaff_prep, work_dir)
    rel_frcmod = os.path.relpath(zaff_frcmod, work_dir)
    rel_pdb = os.path.relpath(prep_pdb, work_dir)

    atype_set = []
    for zc in cfg["zaff_centers"]:
        for row in zc["atom_types"]:
            if row not in atype_set:
                atype_set.append(row)
    atype_block = " ".join(f'{{ "{a}" "{e}" "{h}" }}' for a, e, h in atype_set)
    bond_lines = "\n".join(f"bond mol.{z}.ZN mol.{l}.{atom}" for z, l, atom in bond_specs)
    dna_line = "source leaprc.DNA.OL15" if cfg["dna_chains"] else ""

    # Pass 1: solvate + neutralise to count exact water molecules for 0.15 M KCl
    tleap1 = work_dir / "tleap_pass1.in"
    tleap1.write_text(
        f"""source leaprc.protein.ff14SB
{dna_line}
source leaprc.water.opc
addAtomTypes {{ {atype_block} }}
loadamberparams frcmod.ionslm_126_opc
loadamberprep {rel_prep}
loadamberparams {rel_frcmod}

mol = loadpdb {rel_pdb}
{bond_lines}
check mol
saveamberparm mol dry.prmtop dry.inpcrd
savepdb mol dry.pdb

solvateOct mol OPCBOX {cfg["box_pad_angstrom"]:.1f}
addIonsRand mol K+ 0
addIonsRand mol Cl- 0
saveamberparm mol pass1.prmtop pass1.inpcrd
quit
""",
        encoding="utf-8",
    )
    proc1 = subprocess.run(["tleap", "-s", "-f", tleap1.name], cwd=str(work_dir), capture_output=True, text=True)
    (work_dir / "tleap_pass1.log").write_text(proc1.stdout + "\n" + proc1.stderr, encoding="utf-8")
    if proc1.returncode != 0 or not (work_dir / "pass1.prmtop").exists():
        raise RuntimeError(f"tleap pass 1 failed. Check {work_dir / 'tleap_pass1.log'}")

    p1 = pmd.load_file(str(work_dir / "pass1.prmtop"))
    n_wat = sum(1 for r in p1.residues if r.name == "WAT")
    n_salt_pairs = max(1, round(salt_molarity * n_wat / 55.5))
    print(f"[INFO] Pass 1 waters: {n_wat} -> adding {n_salt_pairs} K+/Cl- pairs for {salt_molarity:.2f} M KCl")

    # Pass 2: build final solvated system with physiological KCl
    tleap2 = work_dir / "tleap_pass2.in"
    solv_prmtop = work_dir / "solv.prmtop"
    solv_inpcrd = work_dir / "solv.inpcrd"
    tleap2.write_text(
        f"""source leaprc.protein.ff14SB
{dna_line}
source leaprc.water.opc
addAtomTypes {{ {atype_block} }}
loadamberparams frcmod.ionslm_126_opc
loadamberprep {rel_prep}
loadamberparams {rel_frcmod}

mol = loadpdb {rel_pdb}
{bond_lines}
solvateOct mol OPCBOX {cfg["box_pad_angstrom"]:.1f}
addIonsRand mol K+ 0
addIonsRand mol Cl- 0
addIonsRand mol K+ {n_salt_pairs} Cl- {n_salt_pairs}
saveamberparm mol solv.prmtop solv.inpcrd
savepdb mol solv.pdb
quit
""",
        encoding="utf-8",
    )
    proc2 = subprocess.run(["tleap", "-s", "-f", tleap2.name], cwd=str(work_dir), capture_output=True, text=True)
    (work_dir / "tleap_pass2.log").write_text(proc2.stdout + "\n" + proc2.stderr, encoding="utf-8")
    if proc2.returncode != 0 or not solv_prmtop.exists():
        raise RuntimeError(f"tleap pass 2 failed. Check {work_dir / 'tleap_pass2.log'}")

    for tmp in ("pass1.prmtop", "pass1.inpcrd"):
        tp = work_dir / tmp
        if tp.exists():
            tp.unlink()

    # Apply ParmEd HMR (solute H -> 3.024 Da, OPC water untouched) for safe 4 fs timestep
    hmr_prmtop = work_dir / "solv_hmr.prmtop"
    print(f"[INFO] Applying ParmEd HMassRepartition (3.024 Da solute H) -> {hmr_prmtop}")
    parm = pmd.load_file(str(solv_prmtop), str(solv_inpcrd))
    orig_sig = [(a.name, a.residue.name) for a in parm.atoms]
    pmd.tools.HMassRepartition(parm, 3.024).execute()
    parm.save(str(hmr_prmtop), overwrite=True)

    parm_check = pmd.load_file(str(hmr_prmtop))
    if [(a.name, a.residue.name) for a in parm_check.atoms] != orig_sig:
        raise RuntimeError("ParmEd reordered atoms in solv_hmr.prmtop relative to solv.prmtop!")

    net_q = sum(a.charge for a in parm.atoms)
    print(f"[OK] Built {hmr_prmtop.name}: {len(parm.atoms):,} atoms, {len(parm.residues):,} residues, net_charge={net_q:+.4f}")
    if abs(net_q) > 0.05:
        raise RuntimeError(f"System net charge is non-zero ({net_q:+.4f})! Check ZAFF charges/ions.")
    return hmr_prmtop, solv_inpcrd


def main() -> None:
    ap = argparse.ArgumentParser(description="Build ZAFF+ff14SB+OL15+OPC+HMR topology for 4XXB or 2AC0_dimer.")
    ap.add_argument("--system", required=True, choices=list(SYSTEM_CONFIGS.keys()))
    ap.add_argument("--mutations", default="WT", help="Comma-separated mutations, e.g. 'A:R280K,B:R280K' or 'B:R326H' or 'WT'.")
    ap.add_argument("--variant", default=None, help="Output variant label (auto-derived if omitted).")
    ap.add_argument("--outdir", required=True, help="Output directory for built topologies and metadata.")
    ap.add_argument("--params-dir", default=None, help="Directory containing ZAFF_H.prep and ZAFF.frcmod.")
    ap.add_argument("--salt", type=float, default=0.15, help="Excess KCl molarity (default: 0.15 M).")
    args = ap.parse_args()

    cfg = SYSTEM_CONFIGS[args.system]
    mut_dict = parse_mutations(args.mutations)
    variant = args.variant or ("WT" if not mut_dict else args.mutations.replace(":", "_").replace(",", "__"))
    outdir = Path(args.outdir).resolve()
    build_dir = outdir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)

    params_dir = (
        Path(args.params_dir).resolve()
        if args.params_dir
        else (Path(__file__).resolve().parent.parent / "params")
    )

    raw_pdb = download_pdb(cfg["pdb_id"], build_dir / f"{cfg['pdb_id']}_raw.pdb")
    zn_coords = extract_zn_coords(raw_pdb, [zc["zn_orig"] for zc in cfg["zaff_centers"]])

    fixed_pdb = build_dir / f"{args.system}_{variant}_fixed.pdb"
    clean_and_mutate_with_pdbfixer(raw_pdb, cfg, mut_dict, fixed_pdb)

    prep_pdb = build_dir / f"{args.system}_{variant}_zaff_input.pdb"
    map_json = build_dir / "residue_map.json"
    bond_specs, meta = prepare_zaff_capped_pdb(fixed_pdb, cfg, zn_coords, prep_pdb, map_json)
    meta["variant"] = variant
    meta["mutations"] = args.mutations
    meta["system_key"] = args.system
    map_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    write_and_run_tleap(prep_pdb, bond_specs, cfg, params_dir, build_dir, salt_molarity=args.salt)
    print(f"[DONE] Build artifacts saved in {build_dir}")


if __name__ == "__main__":
    main()
