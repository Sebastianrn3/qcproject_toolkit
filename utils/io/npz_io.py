import numpy as np
from pathlib import Path


class NPZImageRecorder:
    def __init__(self, folder: Path, atoms):
        self.folder = Path(folder)
        self.atoms = atoms
        self.geoms = []     #list (n_atoms, 3) bohr
        self.energies = []  #list Eh
        self.grads = []     #list of (n_atoms, 3) Eh/bohr
        self.ids = []       #list of int

    def add(self, geom_bohr_2d, energy, grad):
        self.ids.append(len(self.ids))
        self.geoms.append(np.asarray(geom_bohr_2d))
        self.energies.append(energy)
        self.grads.append(np.asarray(grad))

    def save_images(self, subfolder="images", name="images"):
        images_folder = self.folder / subfolder
        images_folder.mkdir(parents=True, exist_ok=True)

        file_path = images_folder / f"{name}.npz"
        np.savez_compressed(
            file_path,
            atoms=self.atoms,
            geoms=np.asarray(self.geoms),
            energies=np.asarray(self.energies),
            grads=np.asarray(self.grads, dtype=float),
            ids=np.asarray(self.ids, dtype=int),
        )
        print("Set of images saved")
        return file_path


def load_all_npz(folder):
    folder = Path(folder)
    results = []
    for file_path in sorted(folder.glob("*.npz")):
        with np.load(file_path, mmap_mode="r") as data:
            results.append({
                "file_path": file_path,
                "atoms": data["atoms"],
                "geoms": data["geoms"],
                "energies": data["energies"],
                "grads": data["grads"],
                "ids": data["ids"],
            })
    return results

def load_all_npz_dict(folder):
    folder = Path(folder)
    out = {}
    for file_path in folder.glob("*.npz"):
        with np.load(file_path, mmap_mode="r") as data:
            out[file_path.name] = {
                "atoms": data["atoms"],
                "geoms": data["geoms"],
                "energies": data["energies"],
                "grads": data["grads"],
                "ids": data["ids"],
            }
    return out

def save_image_npz(params, n_images, name="images", folder = Path("data")/"npz"):
    images_folder = folder/name
    images_folder.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        images_folder,
        atoms=np.asarray(params["atoms"]),
        geoms=np.asarray(params["geoms"], dtype=float),
        energies=np.asarray(params["energies"], dtype=float),
        grads=np.asarray(params["grads"], dtype=float),
        ids=np.array(params["ids"]),
    )
    print(f"{n_images}+2 images saved into {name}.npz")

def load_images_npz(folder):
    folder=Path(folder)
    d = np.load(folder)
    return {
        "atoms": d["atoms"],
        "geoms": d["geoms"],
        "energies=": d["energies"],
        "grads=": d["grads="],
        "ids": d["ids"],
    }

def save_optimize_result_npz(res, folder = Path("data")/"npz", name="opt_result"):
    file_path = folder / f"{name}.npz"
    folder.mkdir(parents=True, exist_ok=True)

    x = np.asarray(res.x, dtype=float)
    jac = np.asarray(res.jac, dtype=float) if getattr(res, "jac", None) is not None else np.array([])

    np.savez_compressed(
        file_path,
        success=bool(res.success),
        status=int(res.status),
        message=str(res.message),
        fun=float(res.fun),
        x=x,
        jac=jac,
        nit=int(getattr(res, "nit", -1)),
        nfev=int(getattr(res, "nfev", -1)),
        njev=int(getattr(res, "njev", -1)),
    )
    print(f"res saved into {file_path}")

def load_optimize_result_npz(name="opt_result",folder=Path("data")/"npz"):
    with np.load(Path(folder) / f"{name}.npz", allow_pickle=True) as data:
        res = {
            "success": bool(data["success"]),
            "status": int(data["status"]),
            "message": str(data["message"]),
            "fun": float(data["fun"]),
            "x": data["x"],
            "jac": data["jac"],
            "nit": int(data["nit"]),
            "nfev": int(data["nfev"]),
            "njev": int(data["njev"]),
        }
    print(res)



