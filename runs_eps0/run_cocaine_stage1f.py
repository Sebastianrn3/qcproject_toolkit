from utils.pipeline.pipeline_protocol import run_multisegment_job

import settings.eps0_jobs.cocaine_stage1f as job
BASE_RUN_NAME = "eps0coc_st1"
INTERPOLATION = "idpp"

POINTS = [
    ("R", job.ES_XYZ),
    ("int1", job.INT1_XYZ),
    ("int2", job.INT2_XYZ),

]

OVERRIDES = {
    "mute_standard_neb": False,
}

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