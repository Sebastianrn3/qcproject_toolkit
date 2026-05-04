from pathlib import Path
import numpy as np


def _expand_bounds(start, stop, min_size, max_idx):
    need = min_size - (stop - start + 1)
    if need > 0:
        start, stop = start - need // 2, stop + (need - need // 2)
        if start < 0:
            stop, start = stop - start, 0
        if stop > max_idx:
            start, stop = start - (stop - max_idx), max_idx
    return max(0, start), min(max_idx, stop)


def _fill_missing(idx, n_images, k_select, acc_dist=None):
    idx = np.unique(idx)
    if len(idx) > 0:
        idx[0], idx[-1] = 0, n_images - 1
        idx = np.unique(idx)

    need = k_select - len(idx)
    if need:
        missing = np.setdiff1d(np.arange(n_images), idx)
        if acc_dist is None:
            idx = np.concatenate([idx, missing[:need]])
        else:
            dists = [(np.min(np.abs(acc_dist[j] - acc_dist[idx])), j) for j in missing]
            idx = np.concatenate([idx, [j for d, j in sorted(dists, reverse=True)[:need]]])

    return np.sort(idx).astype(int)


def parse_relax_tag(file_path):
    tag = Path(file_path).stem.split("_", 1)[1]
    step = int(tag[:-1]) if tag.endswith(("R", "P")) else int(tag.split("of")[0])
    return tag, step


def select_images_by_indices(n_images, k_select, exclude_ends=0):
    if k_select <= 0 or n_images <= 0: return np.array([], dtype=int)
    if exclude_ends == 1: return select_images_by_indices(n_images, k_select + 1)[:-1]
    if exclude_ends == 2: return select_images_by_indices(n_images, k_select + 2)[1:-1]
    if n_images <= k_select: return np.arange(n_images, dtype=int)

    idx = np.round(np.linspace(0, n_images - 1, k_select)).astype(int)
    return _fill_missing(idx, n_images, k_select)


def select_images_by_rmsd(images, k_select, exclude_ends=0):
    n_images = len(images)
    if k_select <= 0 or n_images <= 0: return np.array([], dtype=int)
    if exclude_ends == 1: return select_images_by_rmsd(images, k_select + 1)[:-1]
    if exclude_ends == 2: return select_images_by_rmsd(images, k_select + 2)[1:-1]
    if n_images <= k_select: return np.arange(n_images, dtype=int)

    diffs = images[:-1] - images[1:]
    step_distance = np.sqrt(np.mean(np.sum(diffs ** 2, axis=2), axis=1))

    acc_dist = np.concatenate(([0.0], np.cumsum(step_distance)))

    if acc_dist[-1] < 1e-10:
        return select_images_by_indices(n_images, k_select)

    targets = np.linspace(0.0, acc_dist[-1], k_select)
    idx = [np.argmin(np.abs(acc_dist - t)) for t in targets]
    return _fill_missing(idx, n_images, k_select, acc_dist)


def build_image_groups(folder, k_select, method="rmsd", exclude_ends=0, ranges=None):
    files = sorted(Path(folder).glob("relaxation_*.npz"), key=lambda f: parse_relax_tag(f)[1])
    groups, pool_indices, gid = [], [], 0

    for group_idx, file_path in enumerate(files):
        tag, step = parse_relax_tag(file_path)
        with np.load(file_path, mmap_mode="r") as data:
            geoms, energies, grads = data["geoms"], data["energies"], data["grads"]

        n_total = len(geoms)
        start, stop = 0, n_total - 1

        if ranges and group_idx < len(ranges) and ranges[group_idx]:
            start, stop = max(0, int(ranges[group_idx][0])), min(n_total - 1, int(ranges[group_idx][1]))

        start, stop = _expand_bounds(start, stop, min(k_select, n_total), n_total - 1)

        if stop < start:
            idx = np.array([], dtype=int)
        else:
            exc = 0 if tag.endswith(("R", "P")) else exclude_ends
            if method == "rmsd":
                idx = select_images_by_rmsd(geoms[start:stop + 1], k_select, exc)
            else:
                idx = select_images_by_indices(stop - start + 1, k_select, exc)
            idx += start

        print(f"{file_path.name}: idxs selected: {idx} of {n_total}")

        group = [
            {
                "gid": gid + rank,
                "relax_tag": tag,
                "file_step": step,
                "k_in_file": int(k),
                "select_rank": rank,

                "source_file": file_path.name,
                "source_image": int(k),
                "source_n_images": int(n_total),
                "source_relax_tag": tag,
                "source_relax_step": int(step),

                "geom": geoms[k],
                "energy": float(energies[k]),
                "grad": grads[k],
            }
            for rank, k in enumerate(idx)
        ]

        gid += len(idx)
        groups.append(group)
        pool_indices.append(idx.astype(int))

    return groups, pool_indices


def build_secondary_ranges_from_best(best_full, pool_indices, folder, min_window_size, half_window_expansion=0):
    files = sorted(Path(folder).glob("relaxation_*.npz"), key=lambda f: parse_relax_tag(f)[1])
    n_groups = len(pool_indices)
    ranges = [None] * n_groups

    for g in range(1, n_groups - 1):
        pool = np.asarray(pool_indices[g], dtype=int)
        if len(pool) == 0:
            continue

        rank = int(best_full[g])
        left = int(pool[max(rank - half_window_expansion, 0)])
        right = int(pool[min(rank + half_window_expansion, len(pool) - 1)])

        with np.load(files[g], mmap_mode="r") as data:
            n_total = len(data["geoms"])

        ranges[g] = _expand_bounds(left, right, min_window_size, n_total - 1)

    return ranges