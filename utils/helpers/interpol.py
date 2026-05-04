import numpy as np
import winsound
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.optimize import minimize


def interpolate_linearly(xyz_start: np.ndarray, xyz_end: np.ndarray, n_mid_img: int) -> np.ndarray:
    images = [xyz_start]
    for k in range(1, n_mid_img + 1):
        t = k / (n_mid_img + 1)
        geometry = (1 - t) * xyz_start + t * xyz_end
        images.append(geometry.reshape(-1, 3))
    images.append(xyz_end)
    return np.array(images)


def interpolate_linearly_1d(xyz_start: np.ndarray, xyz_end: np.ndarray, n_mid_img: int) -> np.ndarray:
    images = [xyz_start.ravel()]
    for k in range(1, n_mid_img + 1):
        t = k / (n_mid_img + 1)
        geometry = (1 - t) * xyz_start + t * xyz_end
        images.append(geometry.ravel())
    images.append(xyz_end.ravel())
    return np.array(images)


def interpolate_idpp(xyz_start: np.ndarray, xyz_end: np.ndarray, n_mid_img: int, fixed_atoms: list = None) -> np.ndarray:
    if fixed_atoms is None:
        fixed_atoms = []
    winsound.Beep(400, 7000)
    n_atoms = xyz_start.shape[0]
    images = [xyz_start]
    active_atoms = [i for i in range(n_atoms) if i not in fixed_atoms]

    linear_images = []
    for k in range(1, n_mid_img + 1):
        t = k / (n_mid_img + 1)
        linear_images.append((1 - t) * xyz_start + t * xyz_end)

    d_start = np.linalg.norm(xyz_start[:, None] - xyz_start[None, :], axis=-1)
    d_end = np.linalg.norm(xyz_end[:, None] - xyz_end[None, :], axis=-1)

    for i, guess_xyz in enumerate(linear_images):
        t = (i + 1) / (n_mid_img + 1)
        d_target = (1 - t) * d_start + t * d_end

        def idpp_objective(active_x_flat):
            x = np.copy(guess_xyz)
            x[active_atoms] = active_x_flat.reshape(-1, 3)
            d_current = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
            # Avoid division by zero
            error = np.sum((d_current - d_target) ** 2 / (d_target ** 4 + 1e-6))
            return error

        active_guess = guess_xyz[active_atoms].flatten()
        res = minimize(idpp_objective, active_guess, method='L-BFGS-B')

        final_xyz = np.copy(guess_xyz)
        final_xyz[active_atoms] = res.x.reshape(-1, 3)
        images.append(final_xyz)

    images.append(xyz_end)
    return np.array(images)


def resample_path(geoms: np.ndarray, target_nodes: int) -> np.ndarray:
    geoms_flat = np.array(geoms).reshape(len(geoms), -1)

    diffs = np.diff(geoms_flat, axis=0)
    dists = np.linalg.norm(diffs, axis=1)

    cum_dists = np.insert(np.cumsum(dists), 0, 0.0)
    if cum_dists[-1] == 0:
        return np.array([geoms[0]] * target_nodes)

    cum_dists /= cum_dists[-1]
    target_dists = np.linspace(0, 1, target_nodes)

    interpolator = PchipInterpolator(cum_dists, geoms_flat, axis=0)
    new_geoms_flat = interpolator(target_dists)

    n_atoms = new_geoms_flat.shape[1] // 3
    return new_geoms_flat.reshape(target_nodes, n_atoms, 3)