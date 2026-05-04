from utils.io.xyz_io import write_xyz_series
from utils.statistics.analyse_chain_triplets import analyze_chain_triplets
from pathlib import Path
import re
import numpy as np
from utils.io.npz_io import load_all_npz_dict


def form_combo_report(whole_pull, combo: tuple, cfg, atoms, folder=None):
    if folder is None:
        folder = cfg.analysis_folder

    combo_name = ""
    for i in combo:
        combo_name += f"{i}"
    print(f"combo_name: {combo_name}")

    report_dir = Path(folder) / f"combo_report_{combo_name}_{cfg.jobname}"
    report_dir.mkdir(parents=True, exist_ok=True)

    selected_items = []
    selected_geoms = []

    for group_idx, item_idx in enumerate(combo):
        item = whole_pull[group_idx][item_idx]
        selected_items.append(item)
        selected_geoms.append(item["geom"])


    xyz_name = f"chain_{combo_name}_{cfg.jobname}"

    write_xyz_series(
        atoms=atoms,
        list_of_geoms_bohr=selected_geoms,
        file_name=xyz_name,
        folder=report_dir,
        flatten_all_to_one=True,
    )

    xyz_folder = report_dir / f"{xyz_name}_{len(selected_geoms)}"
    flatten_path = xyz_folder / "flatten.xyz"
    final_xyz_path = report_dir / f"{xyz_name}.xyz"

    if flatten_path.exists():
        final_xyz_path.write_text(flatten_path.read_text(), encoding="utf-8")

        for file in xyz_folder.glob("*.xyz"):
            file.unlink()
        xyz_folder.rmdir()

    data_txt_path = report_dir / f"combo_data_{combo_name}_{cfg.jobname}.txt"
    with data_txt_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"jobname = {cfg.jobname}\n")
        f.write(f"combo   = {combo}\n")
        f.write(f"images  = {len(selected_items)-2}+2\n\n")

        for i, item in enumerate(selected_items):
            f.write("=" * 100 + "\n")
            f.write(f"image {i + 1}/{len(selected_items)}\n")
            f.write(f"group_index = {i}\n")
            f.write(f"combo_index = {combo[i]}\n")
            f.write(f"gid         = {item['gid']}\n")
            f.write(f"relax_tag   = {item['relax_tag']}\n")
            f.write(f"file_step   = {item['file_step']}\n")
            f.write(f"k_in_file   = {item['k_in_file']}\n")
            f.write(f"select_rank = {item['select_rank']}\n")
            f.write(f"energy      = {item['energy']:.16f}\n\n")

            f.write("geometry_bohr:\n")
            np.savetxt(f, np.asarray(item["geom"]), fmt="% .10f")

            f.write("\ngradient_Eh_per_bohr:\n")
            np.savetxt(f, np.asarray(item["grad"]), fmt="% .10f")
            f.write("\n")

    analysis_txt_path = report_dir / f"triplet_analysis_{combo_name}_{cfg.jobname}.txt"
    with analysis_txt_path.open("w", encoding="utf-8", newline="\n") as f:
        analyze_chain_triplets(whole_pull, combo, out=f)

    print(f"Report saved to {report_dir}")
    return report_dir


def _relaxation_sort_key(file_name):
    if file_name.startswith("relaxation_0R"):
        return 0, 0
    m = re.match(r"relaxation_(\d+)of(\d+)\.npz", file_name)
    if m:
        return 1, int(m.group(1))
    m = re.match(r"relaxation_(\d+)P\.npz", file_name)
    if m:
        return 2, int(m.group(1))

    return 99, file_name


def form_relaxations_report(folder, out_path=None):
    npz_dict = load_all_npz_dict(folder)

    file_names = sorted(npz_dict.keys(), key=_relaxation_sort_key)
    n_relax = len(file_names)

    if out_path is None:
        out_path = folder / "all_relaxations_report.txt"
    else:
        out_path = Path(out_path)

    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"folder = {folder}\n")
        f.write(f"n_relaxations = {n_relax}\n\n")

        for relax_idx, file_name in enumerate(file_names, start=1):
            block = npz_dict[file_name]
            geoms = np.asarray(block["geoms"], dtype=float)
            energies = np.asarray(block["energies"], dtype=float)
            grads = np.asarray(block["grads"], dtype=float)

            n_images = len(geoms)

            for img_idx in range(n_images):
                f.write("=" * 100 + "\n")
                f.write(f"relaxation {relax_idx} of {n_relax}    ({file_name})\n")
                f.write(f"image {img_idx + 1} of {n_images}\n")
                f.write(f"E=\n{energies[img_idx]:.16f}\n\n")

                f.write("geom=\n")
                np.savetxt(f, geoms[img_idx], fmt="% .10f")

                f.write("\ngrad=\n")
                np.savetxt(f, grads[img_idx], fmt="% .10f")
                f.write("\n")

    print(f"Relaxation report saved to {out_path}")
    return out_path

def print_best_chain_image_positions(pool, best_full, npz_dict, inner_only=True):
    parts = []

    start_g = 1 if inner_only else 0
    end_g = len(pool) - 1 if inner_only else len(pool)

    for g in range(start_g, end_g):
        chosen_rank = best_full[g]
        item = pool[g][chosen_rank]

        k_in_file = int(item["k_in_file"])
        relax_tag = item["relax_tag"]

        filename = f"relaxation_{relax_tag}.npz"
        total_images = len(npz_dict[filename]["geoms"])

        parts.append(f"{relax_tag}: {k_in_file + 1}/{total_images}")

    text = ", ".join(parts)
    print("Best chain image positions:")
    print(text)
    return text