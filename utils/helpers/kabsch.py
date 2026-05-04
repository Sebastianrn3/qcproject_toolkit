import numpy as np
from pathlib import Path
from utils.io.xyz_io import import_xyz, write_xyz


def evaluate_fixed_atoms_stability(
    xyz_path1,#anchor
    xyz_path2, #rotated
    fixed_atoms_0based,
    save_aligned_xyz=False,
    folder = None,
    filename = None,
):
    r_atoms, r_xyz_raw = import_xyz(xyz_path1)
    p_atoms, p_xyz_raw = import_xyz(xyz_path2)

    if len(r_atoms) != len(p_atoms):
        raise ValueError("XYZ files have different number of atoms")
    print(r_atoms, p_atoms)
    if list(r_atoms) != list(p_atoms):
        print("Atom order/types differ between XYZ files")

    fixed1 = r_xyz_raw[fixed_atoms_0based]
    fixed2 = p_xyz_raw[fixed_atoms_0based]

    # before alignment
    diff_raw = np.linalg.norm(fixed1 - fixed2, axis=1)
    rmsd_raw = np.sqrt(np.mean(diff_raw ** 2))
    max_diff_raw = np.max(diff_raw)

    # centroids
    centroid1 = np.mean(fixed1, axis=0)
    centroid2 = np.mean(fixed2, axis=0)

    f1_centered = fixed1 - centroid1
    f2_centered = fixed2 - centroid2

    # Kabsch
    H = f1_centered.T @ f2_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # reflection fix
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # align ONLY fixed atoms for diagnostics
    fixed2_aligned = f2_centered @ R + centroid1

    diff_aligned = np.linalg.norm(fixed1 - fixed2_aligned, axis=1)
    rmsd_aligned = np.sqrt(np.mean(diff_aligned ** 2))
    max_diff_aligned = np.max(diff_aligned)

    # IMPORTANT: transform for ALL atoms of xyz2
    t = centroid1 - centroid2 @ R
    p_xyz_aligned = p_xyz_raw @ R + t

    print(f"FIXED ATOMS 0BASED:{fixed_atoms_0based} ---")
    print(f"RMSD before:      {rmsd_raw:.6f} bohr")
    print(f"Max diff before:  {max_diff_raw:.6f} bohr")
    print(f"RMSD after:       {rmsd_aligned:.6f} bohr")
    print(f"Max diff after:   {max_diff_aligned:.6f} bohr")

    if rmsd_aligned > 0.09:
        print("DIFFERENCE IS NOTICEABLE - ATTENTION NEEDED")

    for idx, d in zip(fixed_atoms_0based, diff_aligned):
        print(f"atom {idx}: {d:.6f} bohr")

    if save_aligned_xyz:
        write_xyz(
            p_atoms,
            p_xyz_aligned,
            file_name=filename,
            path=folder
        )
        print(f"Aligned xyz saved to: {folder / filename}")
    return