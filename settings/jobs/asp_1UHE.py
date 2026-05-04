# of l-Asparagine a-decarboxylase 135 atoms model

from pathlib import Path
from settings.config import JOBS_DIR, MOPAC_EXE_PATH
from settings.models import JobConfig

#folder name under data/jobs/
JOB_ID = "aspartate_decarb"
JOB_DIR = JOBS_DIR / JOB_ID

#template subfolders
REFERENCES_DIR  = JOB_DIR / "00_references"
INPUT_DIR = JOB_DIR / "01_inputs"
GEOMETRIES_DIR = JOB_DIR / "02_geometries"
OPT_DIR   = JOB_DIR / "03_opt"
NEB_DIR   = JOB_DIR / "04_neb"
ANALYSIS_DIR   = JOB_DIR / "05_analysis"

#raw inputs model 3
R_XYZ_RAW = REFERENCES_DIR / "inputs" / "model3_r.xyz"
P_XYZ_RAW = REFERENCES_DIR / "inputs" / "model3_p.xyz"

#raw inputs model 4.1
# R_XYZ = INPUT_DIR / "model4_2_r.xyz"
# P_XYZ = INPUT_DIR / "model4_2_p.xyz"

#Per-job MOPAC working directory
MOPAC_WORKDIR = OPT_DIR / "mopac"

CHARGE = 0
UNPAIRED_ELECTRONS = 0

FIXED_ATOMS_1BASED = [1, 13, 5, 27, 52, 40, 21, 39, 22, 44] #3
RIGID_GROUPS_1BASED = []

RIGID_GROUPS_1BASED0 =  [
    [*range(1, 5), *range(67, 76), 120],#
    [*range(5, 13), *range(76, 83), 133],
    [*range(13, 20), *range(83, 85), 131, 134],
    [*range(22, 27), *range(86, 93), 119],
    [*range(27, 40), *range(93, 102), 122, 127, 128, 132],#
    [*range(40, 53), *range(102, 110), 121, 123, 126, 129, 130],
    [117, 118, 65] #water
]

# [*range(20, 22), *range(53, 65), 66, 85, *range(110, 117), 124, 125, 135],
FIXED_ATOMS_0BASED = [i - 1 for i in FIXED_ATOMS_1BASED]
RIGID_GROUPS_0BASED = [[i - 1 for i in g] for g in RIGID_GROUPS_1BASED]


CFG = JobConfig(
    jobname=JOB_ID,
    charge=CHARGE,
    unpaired_electrons=UNPAIRED_ELECTRONS,

    fixed_atoms=FIXED_ATOMS_0BASED,
    rigid_groups=None, #RIGID_GROUPS_0BASED,

    mopac_path=MOPAC_WORKDIR,
    mopac_exe=MOPAC_EXE_PATH,

    inputs_folder=INPUT_DIR,
    geometries_folder=GEOMETRIES_DIR,
    opt_folder=OPT_DIR,
    neb_folder=NEB_DIR,
    analysis_folder=ANALYSIS_DIR,
)