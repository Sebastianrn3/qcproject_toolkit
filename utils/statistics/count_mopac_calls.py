import numpy as np
from pathlib import Path


def count_calls_in_raw_folder(raw_folder_path: Path) -> int:
    if not raw_folder_path.exists() or not raw_folder_path.is_dir():
        return 0

    total_calls = 0
    for npz_file in raw_folder_path.glob("*.npz"):
        try:
            with np.load(npz_file, allow_pickle=True) as data:
                if "energies" in data:
                    total_calls += len(data["energies"])
                elif "energies_abs" in data:
                    total_calls += len(data["energies_abs"])
        except Exception as e:
            print(f"   [!] Failed to read {npz_file.name}: {e}")

    return total_calls


def report_mopac_calls_postfactum(opt_folder: Path, run_name: str):
    print(f"\n{'=' * 50}")
    print(f" RETROSPECTIVE MOPAC CALLS COUNTER")
    print(f" Run: {run_name}")
    print(f"{'=' * 50}")

    wave1_raw = opt_folder / f"{run_name}_wave1_raw"
    wave1_calls = count_calls_in_raw_folder(wave1_raw)
    if wave1_calls > 0:
        print(f"➔ Wave 1: {wave1_calls} calls")
    else:
        print(f"➔ Wave 1: Folder not found")

    w1_super_raw = opt_folder / f"{run_name}_wave1superspline_raw"
    if w1_super_raw.exists():
        w1_super_calls = count_calls_in_raw_folder(w1_super_raw)
        print(f"➔ Wave 1 Superspline: {w1_super_calls} calls (atotal: {w1_super_calls})")

    w2_simple_raw = opt_folder / f"{run_name}_wave2simple_raw"
    if w2_simple_raw.exists():
        w2_simple_calls = count_calls_in_raw_folder(w2_simple_raw)
        print(
            f"➔ Wave 2 Simple: {w2_simple_calls} calls (atotal: {wave1_calls + w2_simple_calls})")

    w2_super_raw = opt_folder / f"{run_name}_wave2_split_raw"
    if w2_super_raw.exists():
        w2_super_calls = count_calls_in_raw_folder(w2_super_raw)
        print(
            f"➔ Wave 2 Superspline: {w2_super_calls} вызовов (atotal: {wave1_calls + w2_super_calls})")

    print(f"{'=' * 50}\n")