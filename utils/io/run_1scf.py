import numpy as np
import subprocess

from settings.config import (
    ANGSTROM_PER_BOHR,
    MOPAC_EXE_PATH,
    MOPAC_PATH,
    HARTREE_PER_KCALMOL,
)


def _workdir(cfg):
    return getattr(cfg, "mopac_path", MOPAC_PATH)


def main_mopac(atoms, geometry, cfg):
    create_mopac_input(atoms, geometry, cfg)
    run_mopac_exe(cfg)
    return parse_mopac_output(cfg)


def create_mopac_input(atoms, geometry, cfg):
    folder = _workdir(cfg)
    folder.mkdir(parents=True, exist_ok=True)

    geometry = geometry * ANGSTROM_PER_BOHR

    input_text = (
        f"PM7 UHF 1SCF XYZ GRADIENTS AUX DCART GEO-OK "
        f"MS={cfg.unpaired_electrons / 2:g} CHARGE={cfg.charge}\n"
        f"{cfg.jobname}\n\n"
    )

    for n in range(len(atoms)):
        x = geometry[n]
        input_text += f"{atoms[n]} {x[0]} {x[1]} {x[2]}\n"

    with open(folder / f"{cfg.jobname}.mop", "w") as f:
        f.write(input_text)


def run_mopac_exe(cfg):
    folder = _workdir(cfg)
    mopac_input = folder / f"{cfg.jobname}.mop"

    subprocess.run(
        [str(MOPAC_EXE_PATH), mopac_input.name],
        cwd=str(folder),
        text=True,
        stdout=subprocess.DEVNULL,
    )


def parse_mopac_output(cfg):
    folder = _workdir(cfg)

    with open(folder / f"{cfg.jobname}.aux") as f:
        output = f.read()

    read_from = output.find("HEAT_OF_FORMATION:KCAL/MOL=")
    read_to = output[read_from:].find("\n") + read_from
    E_kcal_mol = float(output[read_from:read_to].replace("D", "E").split("=")[1])
    E_Eh = E_kcal_mol * HARTREE_PER_KCALMOL

    read_from = output.find("GRADIENTS:KCAL/MOL/ANGSTROM")
    read_to = output[read_from:].find("OVERLAP_MATRIX") + read_from

    gradients_kcal_per_A = np.array(output[read_from:read_to].split()[1:], dtype=float)
    gradients_Eh_per_bohr = gradients_kcal_per_A * HARTREE_PER_KCALMOL * ANGSTROM_PER_BOHR

    return E_Eh, gradients_Eh_per_bohr.reshape(-1, 3)


def main_mopac_1d(atoms, geometry, cfg):
    E, g = main_mopac(atoms, geometry.reshape(-1, 3), cfg)
    return E, g.ravel()