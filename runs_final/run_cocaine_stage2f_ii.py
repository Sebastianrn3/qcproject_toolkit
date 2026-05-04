import settings.jobs.cocaine_stage2f as job
from utils.pipeline.pipeline_protocol import run_multisegment_job

BASE_RUN_NAME = "charming_cocaF_stage2_ii"
INTERPOLATION = "linear"

POINTS = [
    ("int2", job.INT2_XYZ),
    ("int3", job.INT3_XYZ),
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