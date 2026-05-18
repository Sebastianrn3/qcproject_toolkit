import numpy as np
from pathlib import Path
from settings.config import ANGSTROM_PER_BOHR


def import_xyz(path: Path):
    with open(path) as f:
        xyz = f.read().strip()
    xyz_lines = xyz.split("\n")
    atoms, geometry = [], []
    for line in range(2, len(xyz_lines)):
        c = xyz_lines[line].split()
        atoms.append(c[0])
        geometry.extend(c[1:])
    geometry = np.array(geometry, dtype=float).reshape(-1, 3)/ANGSTROM_PER_BOHR
    return atoms, geometry

def import_xyz_series(folder: Path):
    folder = Path(folder)
    files = sorted(folder.glob("*.xyz"), key=lambda p: p.name)

    atoms0, geom0 = import_xyz(files[0])
    n_atoms = geom0.shape[0]

    geometries = [geom0]
    for file in files[1:]:
        atoms, geom = import_xyz(file)
        if atoms != atoms0 or geom.shape[0] != n_atoms:
            raise ValueError("Sets of atoms does not match")
        geometries.append(geom)

    return atoms0, np.stack(geometries, axis=0)


def write_xyz(atoms, x_bohr, file_name, path=None):
    if path is not None:
        path = path / f"{file_name}.xyz"
    else:
        path = f"{file_name}.xyz"
    x_ang = x_bohr * ANGSTROM_PER_BOHR
    n = len(atoms)
    new_xyz = f"{n}\n\n"
    for i in range(n):
        new_xyz += atoms[i] + f" {x_ang[i][0]} {x_ang[i][1]} {x_ang[i][2]}\n"
    with open(path, "w") as f:
        f.write(new_xyz)


def write_xyz_series(
        atoms,
        list_of_geoms_bohr,
        file_name,
        folder,
        flatten_all_to_one = True,
):
    folder_path = Path(folder) / f"{file_name}_{len(list_of_geoms_bohr)}"
    folder_path.mkdir(parents=True, exist_ok=True)

    for image_nr, image in enumerate(list_of_geoms_bohr):
        file_path = folder_path / f"{file_name}_{image_nr:02d}.xyz"
        write_xyz(atoms, image, file_name=file_path)

    if flatten_all_to_one:
        flatten_xyz(folder_path)

def flatten_xyz(folder_path: Path, out_name="flatten.xyz"):  # flatten used for animations
    out_path = folder_path / out_name
    xyz_files = sorted(
        [p for p in folder_path.glob("*.xyz") if p.name != out_name],
        key=lambda p: p.name
    )

    with out_path.open("w", newline="\n") as out:
        for f in xyz_files:
            out.write(f.read_text())

    return out_path


def import_multiframe_xyz(path: Path): #load ase result
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')

    if not lines or lines == ['']:
        return [], np.array([])

    n_atoms = int(lines[0].strip())

    lines_per_frame = n_atoms + 2
    n_frames = len(lines) // lines_per_frame

    atoms = []
    geometries = []

    for i in range(n_frames):
        start = i * lines_per_frame
        frame_coords = []

        is_first_frame = (i == 0)

        for j in range(2, 2 + n_atoms):
            parts = lines[start + j].split()
            if is_first_frame:
                atoms.append(parts[0])
            frame_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

        geometries.append(frame_coords)

    geometry_array = np.array(geometries, dtype=float)

    return atoms, geometry_array