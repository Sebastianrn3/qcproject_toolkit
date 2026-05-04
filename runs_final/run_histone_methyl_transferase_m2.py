import settings.jobs.histone_methyl_transferase_m2 as job
from utils.pipeline.pipeline_protocol import run_multisegment_job

BASE_RUN_NAME = "charming_histone2"
INTERPOLATION = "linear"

POINTS = [
    ("R", job.R_XYZ),
    ("P", job.P_XYZ),
]

OVERRIDES = {}

FINAL_STATES = run_multisegment_job(
    job=job,
    base_run_name=BASE_RUN_NAME,
    points=POINTS,
    interpolation=INTERPOLATION,
    overrides={},
)


FINAL_STATE = {
    "name": f"{BASE_RUN_NAME}_all",
    "multi": True,
    "states": FINAL_STATES,
}