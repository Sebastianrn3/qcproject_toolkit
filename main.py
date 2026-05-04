import importlib
import time
import sys
import winsound
import traceback
from pathlib import Path
from datetime import datetime

from utils.statistics.plotgen import generate_word_report

script_start = time.perf_counter()

RUNS = {
    "asp0eps8": "runs_final.run_aspartate_decarboxylase_m0eps8",
    "asp1": "runs_final.run_aspartate_decarboxylase_m1",
    "his0": "runs_final.run_histone_methyl_transferase_m0",
    "hal0": "runs_final.run_haloalcohol_dehalogenase_m0",
    "hal1": "runs_final.run_haloalcohol_dehalogenase_m1",
    "hal2": "runs_final.run_haloalcohol_dehalogenase_m2",

    "his1": "runs_final.run_histone_methyl_transferase_m1",
    "his2": "runs_final.run_histone_methyl_transferase_m2",
    "his3": "runs_final.run_histone_methyl_transferase_m3",

    "asp2": "runs_final.run_aspartate_decarboxylase_m2",
    "asp3": "runs_final.run_aspartate_decarboxylase_m3",

    "cocaf_1ri": "runs_final.run_cocaine_stage1f_ri",
    "cocaf_1ii": "runs_final.run_cocaine_stage1f_ii",
    "cocaf_2ii": "runs_final.run_cocaine_stage2f_ii",
    "cocaf_2ip": "runs_final.run_cocaine_stage2f_ip",

    "acetylene0": "runs_final.run_acetylene_m0",
}

class DualLogger:
    def __init__(self, terminal_stream, log_file_handle):
        self.terminal = terminal_stream
        self.log = log_file_handle

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main(run_name):
    run_start = time.perf_counter()

    base_dir = Path(__file__).resolve().parent
    logs_dir = base_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    log_file_path = logs_dir / f"{run_name}-{timestamp}.txt"

    log_file = open(log_file_path, "a", encoding="utf-8")

    logger_out = DualLogger(sys.stdout, log_file)
    logger_err = DualLogger(sys.stderr, log_file)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = logger_out
    sys.stderr = logger_err

    try:
        print(f"\n{'=' * 50}")
        print(f"--- Running calculation: {run_name} ---")
        print(f"--- Log file: {log_file_path.name} ---")
        print(f"{'=' * 50}\n")

        modname = RUNS.get(run_name)
        if not modname:
            print(f"Err: {run_name} not found in RUNS", file=sys.stderr)
            return 1

        if modname in sys.modules:
            mod = sys.modules[modname]
            importlib.reload(mod)
        else:
            mod = importlib.import_module(modname)

        final_state = getattr(mod, "FINAL_STATE", None)

        if final_state is None:
            print(f"Warning: {run_name} did not provide FINAL_STATE")
            return None

        run_end = time.perf_counter()
        elapsed = run_end - run_start
        print(f"\n{'=' * 50}")
        print(f"Calculation {run_name} finished.")
        print(f"Time wasted (this run): {elapsed:.2f} s")
        print(f"{'=' * 50}\n")

        return final_state

    except Exception as e:
        err_traceback = traceback.format_exc()

        print(f"\nCrit err {run_name}:\n{err_traceback}", file=sys.stderr)

        failures_path = base_dir / "FAILURES_REPORT.txt"
        with open(failures_path, "a", encoding="utf-8") as f:
            time_err = datetime.now().strftime("%Y-%m-%d %H:%M")
            f.write(f"[{time_err}] [{run_name}] CRIT ERRR (PYTHON):\n{err_traceback}\n")
            f.write("-" * 50 + "\n")
        return 1

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

if __name__ == "__main__":
    all_results = []

    tasks = [
        "asp0eps8",
        #"asp1",
        # "his0",
        # "hal0",
        # "hal1",
        # "hal2",
        # "his1",
        "his2",
        # "his3",
        # # "asp3",
        # "cocaf_2ii",
        "cocaf_2ip",
        # "asp2",
        "hal0"
    ]

    for t in tasks:
        try:
            print(f"\nSTARTING TASK: {t}")
            final_state = main(t)

            if isinstance(final_state, dict):
                all_results.append(final_state)
                winsound.Beep(900, 400)
            elif final_state == 1:
                print(f"Task {t} failed. Skipping report input.")
                winsound.Beep(300, 1200)
            else:
                print(f"Task {t} finished, but no FINAL_STATE was returned.")
                winsound.Beep(500, 700)

        except Exception as e:
            print(f"Task {t} failed with error: {e}")

    if all_results:
        print("\n>>> Generating master DOCX report for all tasks...")
        winsound.Beep(1000, 300)
        winsound.Beep(1200, 300)
        winsound.Beep(1400, 600)
        generate_word_report(all_results, "Full_Project_Summary.docx")
    else:
        print("No successful results to report.")
        winsound.Beep(1400, 600)