import winsound
import numpy as np

from utils.helpers.interpol import interpolate_linearly, interpolate_idpp
from utils.pipeline.optimize_masked import sci_minimize_multi
from utils.io.xyz_io import import_xyz, write_xyz, write_xyz_series


def _relax_chain(atoms, geom_set, cfg, fixed, rigid_groups, npz_sub, xyz_name, out_prefix, max_scf_calls=None):
    geom_set = np.asarray(geom_set, dtype=float)
    if geom_set.ndim != 3 or len(geom_set) < 2:
        raise ValueError("geom_set must have shape (n_images, n_atoms, 3) and len >= 2")

    cfg.geometries_folder.mkdir(parents=True, exist_ok=True)
    write_xyz_series(atoms, geom_set, xyz_name, cfg.geometries_folder, flatten_all_to_one=True)

    total, res_energies, all_fixed = len(geom_set), [], list(range(len(atoms)))

    for i, geom in enumerate(geom_set):
        print(f"Relaxing image {i + 1}/{total-2}")
        is_end = (i == 0 or i == total - 1)
        nr = "0R" if i == 0 else (f"{total - 1}P" if is_end else f"{i}of{total - 2}")

        min_geom, res = sci_minimize_multi(
            atoms=atoms, x0_bohr=geom, cfg=cfg, fixed_atoms=all_fixed if is_end else fixed,
            rigid_groups=rigid_groups, npz_record=True, nr=nr, npz_subfolder=npz_sub, max_scf_calls=max_scf_calls
        )
        res_energies.append(res.fun)

        if not res.success and not is_end:
            msg = f"OPTIMIZATION FAILED: {cfg.jobname} ({nr}). CAUSE: {getattr(res, 'message', 'Unknown')}\n"
            print(msg)
            with open("FAILURES_REPORT.txt", "a", encoding="utf-8") as f:
                f.write(msg)
            #winsound.Beep(400, 700)

        write_xyz(atoms, min_geom, f"{out_prefix}_{i}", cfg.geometries_folder)

    print("All relaxations done:", res_energies)


def run_and_record_given_images_relaxation(
        atoms, geom_set, cfg, fixed_atoms,
        rigid_groups=None, npz_subfolder=None,
        xyz_folder_name=None,
        max_scf_calls=None,
):
    _relax_chain(
        atoms, geom_set, cfg, fixed_atoms, rigid_groups,
        npz_sub=npz_subfolder or f"{cfg.jobname}_wave2_raw",
        xyz_name=xyz_folder_name or f"{cfg.jobname}_wave2_interpolated",
        out_prefix=f"{cfg.jobname}_wave2",
        max_scf_calls=max_scf_calls,
    )


def run_and_record_interpolated_images_relaxation(
        reactant_xyz_path, product_xyz_path, interpolation_method, n_interpolated,
        cfg, fixed_atoms, rigid_groups=None, npz_subfolder=None, record_interpolation_only=False, max_scf_calls=None,
):
    r_atoms, r_geom = import_xyz(reactant_xyz_path)
    p_atoms, p_geom = import_xyz(product_xyz_path)
    assert np.array_equal(r_atoms, p_atoms), "Reactant and product atoms mismatch!"

    if interpolation_method == "linear":
        geom_set = interpolate_linearly(r_geom, p_geom, n_interpolated)
    elif interpolation_method == "idpp":
        geom_set = interpolate_idpp(r_geom, p_geom, n_interpolated, fixed_atoms)
    else:
        raise ValueError(f"Invalid interpolation method: {interpolation_method}")

    if record_interpolation_only:
        return geom_set

    _relax_chain(
        r_atoms, geom_set, cfg, fixed_atoms, rigid_groups,
        npz_sub=npz_subfolder or f"{cfg.jobname}_raw",
        xyz_name=f"{cfg.jobname}_wave1_interpolated",
        out_prefix=cfg.jobname,
        max_scf_calls=max_scf_calls,
    )