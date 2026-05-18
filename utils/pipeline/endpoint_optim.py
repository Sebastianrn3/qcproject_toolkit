from pathlib import Path
from math import sqrt

import numpy as np

from settings.config import SHIFT_LIMIT_ANG, ANGSTROM_PER_BOHR
from utils.helpers.kabsch import evaluate_fixed_atoms_stability
from utils.io.xyz_io import import_xyz, write_xyz
from utils.pipeline.optimize_masked import sci_minimize_multi


def stage_0_align(state):
    cfg = state["base_cfg"]

    keys = [
        ("xyz_start_raw", "xyz_start"),
        ("xyz_int1_raw", "xyz_int1"),
        ("xyz_int2_raw", "xyz_int2"),
        ("xyz_int3_raw", "xyz_int3"),
        ("xyz_int4_raw", "xyz_int4"),
        ("xyz_end_raw", "xyz_end")
    ]

    chain = []
    for raw_k, ready_k in keys:
        if state.get(raw_k) is not None and state.get(ready_k) is not None:
            chain.append({
                "raw": Path(state[raw_k]),
                "ready": Path(state[ready_k])
            })

    if len(chain) < 2:
        print("FAILED: Not enough structures to align.")
        return

    ref_raw = chain[0]["raw"]
    aligned_paths = [ref_raw]

    align_atoms = state.get("align_atoms", cfg.fixed_atoms)

    #align
    for i in range(1, len(chain)):
        tgt_raw = chain[i]["raw"]
        tgt_ready = chain[i]["ready"]

        aligned_name = f"{tgt_raw.stem}_aligned"

        evaluate_fixed_atoms_stability(
            ref_raw,
            tgt_raw,
            align_atoms,
            save_aligned_xyz=True,
            folder=tgt_ready.parent,
            filename=aligned_name,
        )
        aligned_paths.append(tgt_ready.parent / f"{aligned_name}.xyz")

    results = []

    #optimize
    for i, (path_to_opt, item) in enumerate(zip(aligned_paths, chain)):
        ready_path = item["ready"]

        atoms, geom = import_xyz(path_to_opt)
        geom_opt, res, cycle_results = relax_endpoint_cycles(
            atoms=atoms,
            geom=geom,
            cfg=cfg,
            state=state,
            ready_path=ready_path,
        )

        write_xyz(atoms, geom_opt, ready_path.with_suffix(""))
        results.append(res)

        if not res.success:
            print(f"FAILED TO OPTIMIZE: {ready_path.name}")
    ref_energy = results[0].fun
    print("\n--- Endpoint & Intermediate Optimizations Report ---")
    for i, item in enumerate(chain):
        diff_hartree = results[i].fun - ref_energy
        diff_kcal = diff_hartree * 627.509

        if i == 0:
            print(f"[{i}] {item['ready'].stem: <15} | Base Energy: {results[i].fun:.6f} Eh")
        else:
            print(f"[{i}] {item['ready'].stem: <15} | Rel: {diff_hartree:+.6f} Eh ({diff_kcal:+.2f} kcal/mol)")

    print("----------------------------------------------------\n")


def opt_both_endpoints(
        R_path: Path,
        P_path: Path,
        cfg,
        fixed_atoms,
        rigid_groups=None,
        npz_record=False
):
    r_atoms, r_xyz_raw = import_xyz(R_path)
    p_atoms, p_xyz_raw = import_xyz(P_path)
    assert np.array_equal(r_atoms, p_atoms)

    r_xyz_optimized, res_r = sci_minimize_multi(
        r_atoms,
        r_xyz_raw,
        cfg,
        fixed_atoms,
        rigid_groups,
        npz_record,
        nr="R_optimized"
    )

    p_xyz_optimized, res_p = sci_minimize_multi(
        p_atoms,
        p_xyz_raw,
        cfg,
        fixed_atoms,
        rigid_groups,
        npz_record,
        nr="P_optimized"
    )

    write_xyz(r_atoms, r_xyz_optimized, file_name="R_opt", path=cfg.inputs_folder)
    write_xyz(p_atoms, p_xyz_optimized, file_name="P_opt", path=cfg.inputs_folder)

    if check_fixed_atom_shifts(r_xyz_optimized, p_xyz_optimized, fixed_atoms):
        print(f"Too large >{SHIFT_LIMIT_ANG} shifts found among reactant and product fixed atoms. Try Kabsch...")
        assert False


def optimize_endpoint(
        xyz_path: Path,
        fixed_atoms,
        cfg,
        rigid_groups=None,
        npz_record=False
):
    atoms, geometry_raw = import_xyz(xyz_path)

    geom_new, res = sci_minimize_multi(
        atoms,
        geometry_raw,
        cfg,
        fixed_atoms,
        rigid_groups,
        npz_record,
        nr="optimized",
    )
    write_xyz(atoms, geom_new, file_name="optimized_endpoint", path=cfg.inputs_folder)
    print("Endpoint optimized")

def check_fixed_atom_shifts(r_xyz, p_xyz, fixed_list, shift_limit_ang=SHIFT_LIMIT_ANG):
    r_xyz = np.asarray(r_xyz, dtype=float).reshape(-1, 3)
    p_xyz = np.asarray(p_xyz, dtype=float).reshape(-1, 3)

    limit_bohr = shift_limit_ang / ANGSTROM_PER_BOHR

    fixed0 = [(i-1) for i in fixed_list]
    for atom in fixed0:
        shift = sqrt(
            (r_xyz[atom, 0] - p_xyz[atom, 0]) ** 2 +
            (r_xyz[atom, 1] - p_xyz[atom, 1]) ** 2 +
            (r_xyz[atom, 2] - p_xyz[atom, 2]) ** 2
        )
        if shift > limit_bohr:
            return True
    return False

HARTREE_TO_KCALMOL = 627.5094740631
HARTREE_TO_KJMOL = 2625.49962


def relax_endpoint_cycles(atoms, geom, cfg, state, ready_path):
    endpoint_cycles = int(state.get("endpoint_relax_cycles", 2))
    endpoint_max_scf_calls = state.get("endpoint_max_scf_calls", None)
    stop_kcal = float(state.get("endpoint_relax_stop_kcal", 0.01))

    geom_current = geom
    cycle_results = []

    for cycle in range(endpoint_cycles):
        print(f"\nEndpoint relax cycle {cycle + 1}/{endpoint_cycles}: {ready_path.name}")

        geom_current, res = sci_minimize_multi(
            atoms,
            geom_current,
            cfg,
            fixed_atoms=cfg.fixed_atoms,
            rigid_groups=cfg.rigid_groups,
            npz_record=False,
            nr=f"{ready_path.stem}_cycle{cycle + 1}",
            max_scf_calls=endpoint_max_scf_calls,
        )

        cycle_results.append(res)
        e_eh = float(res.fun)

        print(f"  E[{cycle + 1}] = {e_eh:.10f} Eh")
        print(f"  success = {res.success}")
        print(f"  message = {res.message}")

        if cycle > 0:
            prev_e = float(cycle_results[cycle - 1].fun)
            d_eh = e_eh - prev_e
            d_kcal = d_eh * HARTREE_TO_KCALMOL
            d_kj = d_eh * HARTREE_TO_KJMOL

            print(
                f"  ΔE cycle {cycle}->{cycle + 1} = "
                f"{d_eh:+.10e} Eh = {d_kcal:+.6f} kcal/mol = {d_kj:+.6f} kJ/mol"
            )

            if res.success and abs(d_kcal) < stop_kcal:
                print(f"  Endpoint stable: |ΔE| < {stop_kcal} kcal/mol")
                break

    energies = np.array([float(r.fun) for r in cycle_results])
    rel_kcal = (energies - energies[0]) * HARTREE_TO_KCALMOL

    print("\n  Endpoint relaxation energy history:")
    for j, (e, de) in enumerate(zip(energies, rel_kcal), start=1):
        print(f"    cycle {j}: E = {e:.10f} Eh, ΔE from cycle 1 = {de:+.6f} kcal/mol")

    return geom_current, cycle_results[-1], cycle_results