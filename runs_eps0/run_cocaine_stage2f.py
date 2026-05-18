import settings.eps0_jobs.cocaine_stage2f as job
from utils.pipeline.pipeline_protocol import run_multisegment_job

BASE_RUN_NAME = "eps0coc_st2"
INTERPOLATION = "idpp"

POINTS = [
    ("int2", job.INT2_XYZ),
    ("int3", job.INT3_XYZ),
    ("P", job.PROD_XYZ),
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