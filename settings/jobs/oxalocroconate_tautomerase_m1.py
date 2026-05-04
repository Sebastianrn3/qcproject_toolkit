from settings.config import JOBS_DIR, MOPAC_EXE_PATH
from settings.models import JobConfig

#folder name under data/jobs/
JOB_ID = "oxalocrotonate_tautomerase"
JOB_DIR = JOBS_DIR / JOB_ID

#template subfolders
REFERENCES_DIR  = JOB_DIR / "00_references"
INPUT_DIR = JOB_DIR / "01_inputs"
GEOMETRIES_DIR = JOB_DIR / "02_geometries"
OPT_DIR   = JOB_DIR / "03_opt"
NEB_DIR   = JOB_DIR / "04_neb"
ANALYSIS_DIR   = JOB_DIR / "05_analysis"

#raw inputs
R_XYZ = REFERENCES_DIR / "4OTA" / "model1red" /"model1_r.xyz"
INT_XYZ = REFERENCES_DIR / "4OTA" / "model1red" /"model1_int_reordered.xyz"
P_XYZ = REFERENCES_DIR / "4OTA" / "model1red" /"model1_p_reordered.xyz"

#Per-job MOPAC working directory
MOPAC_WORKDIR = OPT_DIR / "mopac"

CHARGE = +1
UNPAIRED_ELECTRONS = 0

FIXED_ATOMS_1BASED = [7, 74, 33, 58, 40, 49, 45, 82, 66]
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