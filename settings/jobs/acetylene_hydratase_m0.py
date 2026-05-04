from settings.config import JOBS_DIR, MOPAC_EXE_PATH
from settings.models import JobConfig

#folder name under data/jobs/
JOB_ID = "acetilene_hydratase"
JOB_DIR = JOBS_DIR / JOB_ID

#template subfolders
REFERENCES_DIR  = JOB_DIR / "00_references"
INPUT_DIR = JOB_DIR / "01_inputs"
GEOMETRIES_DIR = JOB_DIR / "02_geometries"
OPT_DIR   = JOB_DIR / "03_opt"
NEB_DIR   = JOB_DIR / "04_neb"
ANALYSIS_DIR   = JOB_DIR / "05_analysis"

#raw inputs
R_XYZ =   REFERENCES_DIR / "5E7Z" /"model_0_r.xyz"
INT1_XYZ =   REFERENCES_DIR / "5E7Z" / "model_0_int1.xyz"
INT2_XYZ =   REFERENCES_DIR / "5E7Z" / "model_0_int2.xyz"
INT3_XYZ =   REFERENCES_DIR / "5E7Z" / "model_0_int3.xyz"
INT4_XYZ =   REFERENCES_DIR / "5E7Z" / "model_0_int4.xyz"
P_XYZ =   REFERENCES_DIR / "5E7Z" / "model_0_p.xyz"

# """


#Per-job MOPAC working directory
MOPAC_WORKDIR = OPT_DIR / "mopac"

CHARGE = -1
UNPAIRED_ELECTRONS = 0

FIXED_ATOMS_1BASED = [1, 5, 10, 13, 22, 23, 29, 38, 57, 58]
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