from utils.pipeline.pipeline_protocol import run_multisegment_job

import settings.eps0_jobs.acetylene_hydratase_m0 as job
BASE_RUN_NAME = "eps0acetylene_m0"
INTERPOLATION = "idpp"

POINTS = [
    ("R", job.R_XYZ),
    ("int1", job.INT1_XYZ),
    ("int2", job.INT2_XYZ),
    ("int3", job.INT3_XYZ),
    ("int4", job.INT4_XYZ),
    ("P", job.P_XYZ),
]

OVERRIDES = {}

FINAL_STATES = run_multisegment_job(
    job=job,
    base_run_name=BASE_RUN_NAME,
    points=POINTS,
    interpolation=INTERPOLATION,
    overrides=OVERRIDES,
)

FINAL_STATE = {
    "name": f"{BASE_RUN_NAME}_all",
    "multi": True,
    "states": FINAL_STATES,
}