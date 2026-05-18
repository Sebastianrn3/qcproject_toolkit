from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms, io
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.mep import NEB
from ase.optimize import FIRE, LBFGS
from matplotlib import pyplot as plt

from settings.config import ANGSTROM_PER_BOHR
from utils.helpers.interpol import interpolate_idpp, interpolate_linearly
from utils.helpers.metadata import write_metadata
from utils.io.run_1scf import main_mopac
from utils.io.xyz_io import import_xyz


HARTREE_TO_EV = 27.211386245988
HARTREE_BOHR_TO_EV_ANG = HARTREE_TO_EV / ANGSTROM_PER_BOHR
HARTREE_TO_KCALMOL = 627.5094740631

OPTIMIZERS = {
    "FIRE": FIRE,
    "LBFGS": LBFGS,
}


class MopacASECalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, atoms_list, base_cfg, workdir, label, **kwargs):
        super().__init__(**kwargs)

        self.atoms_list = list(atoms_list)
        self.base_cfg = base_cfg
        self.workdir = Path(workdir)
        self.label = str(label)
        self.n_calls = 0

        self.workdir.mkdir(parents=True, exist_ok=True)

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            atoms = self.atoms

        self.n_calls += 1

        geom_bohr = np.asarray(atoms.get_positions(), dtype=float) / ANGSTROM_PER_BOHR

        local_cfg = replace(
            self.base_cfg,
            mopac_path=self.workdir,
            jobname=self.label,
        )

        energy_eh, grad_eh_bohr = main_mopac(
            self.atoms_list,
            geom_bohr,
            local_cfg,
        )

        self.results["energy"] = float(energy_eh) * HARTREE_TO_EV
        self.results["forces"] = -np.asarray(grad_eh_bohr, dtype=float) * HARTREE_BOHR_TO_EV_ANG


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    raise TypeError(f"Type {type(obj)} is not JSON serializable")


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path = Path(path)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )


def _fixed_atom_indices(cfg) -> list[int]:
    fixed_atoms = getattr(cfg, "fixed_atoms", None)

    if fixed_atoms is None:
        return []

    if isinstance(fixed_atoms, np.ndarray):
        return [int(x) for x in fixed_atoms.tolist()]

    if isinstance(fixed_atoms, (list, tuple, set)):
        return [int(x) for x in fixed_atoms]

    return [int(fixed_atoms)]


def _collect(images, neb_obj) -> dict[str, Any]:
    energies = np.asarray(
        [img.get_potential_energy() for img in images],
        dtype=float,
    ) / HARTREE_TO_EV

    fmax_per_image = np.zeros(len(images), dtype=float)

    if len(images) > 2:
        forces_interior = neb_obj.get_forces().reshape(len(images) - 2, -1, 3)
        fmax_interior = np.sqrt((forces_interior**2).sum(axis=-1)).max(axis=-1)
        fmax_per_image[1:-1] = fmax_interior

    e_min = float(energies.min())
    e_0 = float(energies[0])

    return {
        "energies_abs": energies.tolist(),
        "energies_rel": (energies - e_min).tolist(),
        "energies_rel_start": (energies - e_0).tolist(),
        "barrier": float(energies.max() - e_min),
        "activation": float(energies.max() - e_0),
        "peak_index": int(np.argmax(energies)),
        "start": e_0,
        "end": float(energies[-1]),
        "neb_fmax": float(fmax_per_image.max()),
        "fmax_per_image": fmax_per_image.tolist(),
    }


def _save_final_beads(output_dir: Path, run_name: str, ase_images) -> Path:
    beads_dir = output_dir / f"{run_name}_beads_final"
    beads_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(ase_images):
        io.write(beads_dir / f"image_{i:02d}.xyz", img)

    return beads_dir


def _save_summary_npz(
    output_dir: Path,
    run_name: str,
    atoms,
    initial: dict[str, Any],
    final: dict[str, Any],
    ase_images,
    total_calls: int,
    opt,
    converged: bool,
) -> Path:
    summary_npz = output_dir / f"{run_name}_summary.npz"

    np.savez_compressed(
        summary_npz,
        atoms=np.asarray(atoms),
        initial_energies=np.asarray(initial["energies_abs"], dtype=float),
        final_energies=np.asarray(final["energies_abs"], dtype=float),
        initial_rel=np.asarray(initial["energies_rel"], dtype=float),
        initial_rel_start=np.asarray(initial["energies_rel_start"], dtype=float),
        final_rel=np.asarray(final["energies_rel"], dtype=float),
        final_rel_start=np.asarray(final["energies_rel_start"], dtype=float),
        initial_fmax=float(initial["neb_fmax"]),
        final_fmax=float(final["neb_fmax"]),
        initial_fmax_per_image=np.asarray(initial["fmax_per_image"], dtype=float),
        final_fmax_per_image=np.asarray(final["fmax_per_image"], dtype=float),
        total_mopac_calls=int(total_calls),
        per_image_mopac_calls=np.asarray([img.calc.n_calls for img in ase_images], dtype=int),
        n_neb_cycles=int(opt.nsteps),
        nsteps_taken=int(opt.nsteps),
        converged=bool(converged),
    )

    return summary_npz


def _plot_energy_profile(summary_npz: Path, title: str = "NEB energy profile") -> Path | None:
    summary_npz = Path(summary_npz)

    if not summary_npz.exists():
        return None

    with np.load(summary_npz, allow_pickle=True) as data:
        if "final_rel_start" in data:
            energies_kcalmol = np.asarray(data["final_rel_start"], dtype=float) * HARTREE_TO_KCALMOL
        elif "final_rel" in data:
            energies_kcalmol = np.asarray(data["final_rel"], dtype=float) * HARTREE_TO_KCALMOL
            energies_kcalmol = energies_kcalmol - energies_kcalmol[0]
        else:
            return None

    out_path = summary_npz.with_name(
        summary_npz.stem.replace("_summary", "_energy_profile") + ".png"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(range(len(energies_kcalmol)), energies_kcalmol, marker="o")

    for i, e in enumerate(energies_kcalmol):
        plt.text(i, e, str(i), fontsize=9, ha="left", va="bottom")

    plt.xticks(range(len(energies_kcalmol)))
    plt.xlabel("Bead index")
    plt.ylabel("Relative energy, kcal/mol")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

    return out_path


def _read_existing_n_images(neb_dir: Path, run_name: str) -> int | None:
    summary_npz = neb_dir / f"{run_name}_summary.npz"
    results_json = neb_dir / f"{run_name}_results.json"

    if summary_npz.exists():
        try:
            with np.load(summary_npz, allow_pickle=True) as data:
                if "final_energies" in data:
                    return int(len(data["final_energies"]))
                if "final_rel" in data:
                    return int(len(data["final_rel"]))
        except Exception:
            pass

    if results_json.exists():
        try:
            with results_json.open("r", encoding="utf-8") as f:
                data = json.load(f)

            energies = data.get("final", {}).get("energies_abs")
            if energies is not None:
                return int(len(energies))
        except Exception:
            pass

    return None


def run_ase_neb(
    atoms,
    images,
    cfg,
    output_dir,
    fmax,
    steps,
    k,
    run_name,
    method="improvedtangent",
    optimizer="FIRE",
    climb=False,
):
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = np.asarray(images, dtype=float)

    if images.ndim != 3 or images.shape[2] != 3:
        raise ValueError("images must have shape (n_images, n_atoms, 3) in bohr")

    if len(images) < 3:
        raise ValueError("NEB requires at least 3 images: reactant, intermediate, product")

    optimizer_key = str(optimizer).upper()

    if optimizer_key not in OPTIMIZERS:
        raise ValueError(f"Unknown NEB optimizer: {optimizer}. Available: {sorted(OPTIMIZERS)}")

    ase_images = [
        Atoms(
            symbols=atoms,
            positions=geom_bohr * ANGSTROM_PER_BOHR,
        )
        for geom_bohr in images
    ]

    fixed_atoms = _fixed_atom_indices(cfg)

    if fixed_atoms:
        constraint = FixAtoms(indices=fixed_atoms)

        for img in ase_images:
            img.set_constraint(constraint)

    for i, img in enumerate(ase_images):
        img.calc = MopacASECalculator(
            atoms,
            cfg,
            output_dir / "mopac" / f"img_{i:02d}",
            f"{cfg.jobname}_{i:02d}",
        )

    io.write(output_dir / f"{run_name}_initial_chain.xyz", ase_images)

    neb = NEB(
        ase_images,
        k=float(k),
        method=method,
        climb=climb,
        allow_shared_calculator=False,
    )

    opt = OPTIMIZERS[optimizer_key](
        neb,
        trajectory=str(output_dir / f"{run_name}.traj"),
        logfile=str(output_dir / f"{run_name}.log"),
    )

    print("Initial energies...")
    initial = _collect(ase_images, neb)

    print("Running NEB...")
    converged = opt.run(
        fmax=float(fmax),
        steps=int(steps),
    )

    print("Final energies...")
    final = _collect(ase_images, neb)

    total_calls = int(sum(img.calc.n_calls for img in ase_images))

    io.write(output_dir / f"{run_name}_final_chain.xyz", ase_images)
    beads_dir = _save_final_beads(output_dir, run_name, ase_images)

    summary_npz = _save_summary_npz(
        output_dir=output_dir,
        run_name=run_name,
        atoms=atoms,
        initial=initial,
        final=final,
        ase_images=ase_images,
        total_calls=total_calls,
        opt=opt,
        converged=bool(converged),
    )

    results = {
        "converged": bool(converged),
        "nsteps_taken": int(opt.nsteps),
        "n_neb_cycles": int(opt.nsteps),
        "elapsed_seconds": float(time.time() - t0),
        "total_mopac_calls": total_calls,
        "per_image_mopac_calls": [int(img.calc.n_calls) for img in ase_images],
        "initial": initial,
        "final": final,
        "files": {
            "traj": str((output_dir / f"{run_name}.traj").resolve()),
            "log": str((output_dir / f"{run_name}.log").resolve()),
            "summary_npz": str(summary_npz.resolve()),
            "initial_chain_xyz": str((output_dir / f"{run_name}_initial_chain.xyz").resolve()),
            "final_chain_xyz": str((output_dir / f"{run_name}_final_chain.xyz").resolve()),
            "beads_dir": str(beads_dir.resolve()),
        },
    }

    _save_json(output_dir / f"{run_name}_results.json", results)

    print("=" * 60)
    print(f"Converged: {converged}")
    print(f"Optimizer steps: {opt.nsteps}")
    print(f"MOPAC calls: {total_calls}")
    print(f"Final fmax: {final['neb_fmax']:.6f} eV/Å")
    print(f"Final activation: {final['activation']:.6f} Eh")
    print("=" * 60)

    return results


def _build_standard_guess(PIPELINE_CONFIG, start_path=None, end_path=None):
    atoms, start = import_xyz(start_path or PIPELINE_CONFIG["xyz_start"])
    atoms_end, end = import_xyz(end_path or PIPELINE_CONFIG["xyz_end"])

    if list(atoms) != list(atoms_end):
        raise ValueError("Start and end XYZ files have different atom lists.")

    interp_method = PIPELINE_CONFIG.get("interpolation_method", "idpp").lower()
    n_mid_images = int(PIPELINE_CONFIG["target_neb_beads"]) - 2

    if n_mid_images < 1:
        raise ValueError("target_neb_beads must be at least 3.")

    if interp_method == "linear":
        geom_set = interpolate_linearly(start, end, n_mid_images)
    elif interp_method == "idpp":
        geom_set = interpolate_idpp(
            start,
            end,
            n_mid_images,
            PIPELINE_CONFIG["base_cfg"].fixed_atoms,
        )
    else:
        raise ValueError(f"Unknown interpolation method: {interp_method}")

    return atoms, geom_set


def neb_wrapper(
    PIPELINE_CONFIG,
    myguess=None,
    atoms=None,
    start_path=None,
    end_path=None,
    run_name=None,
):
    run_name = run_name or PIPELINE_CONFIG["name"]
    cfg = PIPELINE_CONFIG["base_cfg"]
    neb_dir = cfg.neb_folder / f"neb_{run_name}"

    results_json = neb_dir / f"{run_name}_results.json"
    summary_npz = neb_dir / f"{run_name}_summary.npz"

    geom_set = None
    method_label = "custom" if myguess is not None else "standard"

    if results_json.exists():
        print(f"NEB already exists: {run_name}")
    else:
        if myguess is not None:
            geom_set = np.asarray(myguess, dtype=float)

            if atoms is None:
                atoms, _ = import_xyz(start_path or PIPELINE_CONFIG["xyz_start"])
        else:
            atoms, geom_set = _build_standard_guess(
                PIPELINE_CONFIG,
                start_path=start_path,
                end_path=end_path,
            )

        run_ase_neb(
            atoms=atoms,
            images=geom_set,
            cfg=cfg,
            output_dir=neb_dir,
            fmax=PIPELINE_CONFIG["neb_fmax"],
            steps=PIPELINE_CONFIG["neb_steps"],
            k=PIPELINE_CONFIG["neb_spring_constant"],
            method=PIPELINE_CONFIG["neb_method"],
            optimizer=PIPELINE_CONFIG["neb_optimizer"],
            run_name=run_name,
        )

    if geom_set is not None:
        n_images = int(len(geom_set))
    else:
        n_images = _read_existing_n_images(neb_dir, run_name)

    extra = {
        "n_images": n_images,
    }

    write_metadata(
        neb_dir,
        state=PIPELINE_CONFIG,
        kind="neb",
        method=method_label,
        run_name=run_name,
        extra=extra,
    )

    _plot_energy_profile(
        summary_npz,
        title=f"NEB energy profile: {run_name}",
    )