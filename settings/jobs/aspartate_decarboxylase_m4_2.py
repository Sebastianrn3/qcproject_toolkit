from settings.config import JOBS_DIR, MOPAC_EXE_PATH
from settings.models import JobConfig

#folder name under data/jobs/
JOB_ID = "aspartate_decarboxylase"
JOB_DIR = JOBS_DIR / JOB_ID

#template subfolders
REFERENCES_DIR  = JOB_DIR / "00_references"
INPUT_DIR = JOB_DIR / "01_inputs"
GEOMETRIES_DIR = JOB_DIR / "02_geometries"
OPT_DIR   = JOB_DIR / "03_opt"
NEB_DIR   = JOB_DIR / "04_neb"
ANALYSIS_DIR   = JOB_DIR / "05_analysis"

#raw inputs
R_XYZ =   REFERENCES_DIR / "model4_2" / "model4_2_r.xyz"
P_XYZ =   REFERENCES_DIR / "model4_2" / "model4_2_p.xyz"

#Per-job MOPAC working directory
MOPAC_WORKDIR = OPT_DIR / "mopac"

CHARGE = 0
UNPAIRED_ELECTRONS = 0

FIXED_ATOMS_1BASED = [5, 1, 64, 11, 67, 33, 45, 62, 54, 50, 27, 28]
FIXED_ATOMS_0BASED = [i - 1 for i in FIXED_ATOMS_1BASED]

CFG = JobConfig(
    jobname=JOB_ID,
    charge=CHARGE,
    unpaired_electrons=UNPAIRED_ELECTRONS,

    fixed_atoms=FIXED_ATOMS_0BASED,
    rigid_groups=None,

    mopac_path=MOPAC_WORKDIR,
    mopac_exe=MOPAC_EXE_PATH,

    inputs_folder=INPUT_DIR,
    geometries_folder=GEOMETRIES_DIR,
    opt_folder=OPT_DIR,
    neb_folder=NEB_DIR,
    analysis_folder=ANALYSIS_DIR,
)