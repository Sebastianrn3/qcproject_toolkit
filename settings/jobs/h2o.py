from settings.models import JobConfig
from settings.config import JOBS_DIR, MOPAC_EXE_PATH

#folder name under data/jobs/
JOB_ID = "h2o"
JOB_DIR = JOBS_DIR / JOB_ID

#template subfolders
REFERENCES_DIR  = JOB_DIR / "00_references"
INPUT_DIR = JOB_DIR / "01_inputs"
GEOMETRIES_DIR = JOB_DIR / "02_geometries"
OPT_DIR   = JOB_DIR / "03_opt"
NEB_DIR   = JOB_DIR / "04_neb"
ANALYSIS_DIR   = JOB_DIR / "05_analysis"

#raw inputs
REACTANT_XYZ = INPUT_DIR / "r.xyz"
PRODUCT_XYZ  = INPUT_DIR / "p.xyz"


CHARGE = 0
UNPAIRED_ELECTRONS = 0

#Per-job MOPAC working directory
MOPAC_WORKDIR = OPT_DIR / "mopac"

FIXED_ATOMS_0BASED = [0]

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


