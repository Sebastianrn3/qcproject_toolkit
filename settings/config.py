from pathlib import Path

SHIFT_LIMIT_ANG = 0.05
EPS = 1e-12

KCALMOL_TO_EV = 0.0433641153087705
EV_TO_HARTREE = 1 / 27.211386245988
HARTREE_PER_KCALMOL = 1 / 627.5094740631
ANGSTROM_PER_BOHR = 0.529177210903

#directories
MOPAC_PATH =  Path.cwd().parents[2]/"mopac_runs"
MOPAC_EXE_PATH = Path.cwd().parents[2]/"mopac/bin/mopac.exe"

ROOT = Path.cwd().parents[1]
DATA_DIR = ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"

NPZ_FOLDER = DATA_DIR / "npz"
XYZ_FOLDER = DATA_DIR / "xyz"
