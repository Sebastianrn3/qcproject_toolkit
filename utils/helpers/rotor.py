import numpy as np

def prepare_rotors(x0_bohr_1d, rigid_groups, anchor_ids):
    x0_bohr = np.asarray(x0_bohr_1d).reshape(-1, 3)
    groups_data = []

    for group in rigid_groups:
        atoms = np.asarray(group)
        anchors_in_group = [i for i in atoms if i in anchor_ids]

        if len(anchors_in_group) == 2: #axis twist
            a0, a1 = anchors_in_group
            moving_atoms = [i for i in atoms if i not in anchors_in_group]

            p0 = x0_bohr[a0].copy()
            p1 = x0_bohr[a1].copy()

            axis_vector = p1 - p0
            n = np.linalg.norm(axis_vector)
            u0 = axis_vector/n

            rel0 = x0_bohr[moving_atoms]- p0

            groups_data.append({
                "atoms": moving_atoms,
                "axis_dimension": "2d",
                "ndof": 1,
                "a0": a0,
                "a1": a1,
                "p0": p0,
                "u0": u0,
                "rel0": rel0,
            })
            continue

        if len(anchors_in_group) == 1: #joint twist/ 3DoF
            anchor_id = anchors_in_group[0]
            moving_atoms = np.array([i for i in group if i != anchor_id])

            anchor0 = x0_bohr[anchor_id].copy()
            rel0 = x0_bohr[moving_atoms] - anchor0

            groups_data.append({
                "atoms": moving_atoms,
                "axis_dimension": "1d",
                "ndof": 3,
                "anchor_id": anchor_id,
                "anchor0": anchor0,
                "rel0": rel0,
            })
            continue
        assert False, "Invalid anchor"
    return groups_data


def build_geom_from_rotors(x, x0_bohr_1d, rotor_data, free_atoms):
    x0_bohr = np.asarray(x0_bohr_1d).reshape(-1, 3)
    full_geom = x0_bohr.copy()

    offset = 0
    for group in rotor_data:
        if group["axis_dimension"] == "1d":
            w = np.asarray(x[offset:offset + 3], float)
            offset += 3
            R = rodrigues_from_vector(w)

            xyz_group = group["anchor0"] + group["rel0"] @ R.T
            full_geom[group["atoms"]] = xyz_group

        elif group["axis_dimension"] == "2d":
            angle = float(x[offset])
            offset += 1
            rel_rot = rotate_around_2d_axis(group["rel0"], group["u0"], angle)

            xyz_group = group["p0"] + rel_rot
            full_geom[group["atoms"]] = xyz_group
        else:
            assert False, f"Invalid anchor axis format"

    xyz_free_atoms = np.asarray(x[offset:]).reshape(-1, 3)
    full_geom[free_atoms] = xyz_free_atoms
    return full_geom.ravel()

def rodrigues_from_vector(w):
    w = np.asarray(w, float)
    teta = np.linalg.norm(w)  # teta = |w|
    if teta < 1e-12:
        return np.eye(3)
    k = w / teta
    kx, ky, kz = k
    K = np.array([
        [0, -kz, ky],
        [kz, 0, -kx],
        [-ky, kx, 0],
    ])  # Kv = k*v
    I = np.eye(3)
    R = I + np.sin(teta) * K + (1 - np.cos(teta)) * (K @ K)
    return R

def rotate_around_2d_axis(vectors, u, angle):
    v = np.asarray(vectors)
    u = np.asarray(u)

    R = (v * np.cos(angle) +
         np.cross(u, v) * np.sin(angle) +
         u * (np.sum(v * u, axis=1, keepdims=True)) * (1 - np.cos(angle)))
    return R

def pack_gradients_multi_rotor(G_full, full_bohr, rotor_data,free_atoms):
    G_full = np.asarray(G_full).reshape(-1, 3)
    coords = np.asarray(full_bohr).reshape(-1, 3)

    G_parts = []

    for group in rotor_data:
        group_atoms = group["atoms"]
        axis_dimension = group["axis_dimension"]

        if axis_dimension == "1d":
            anchor_id = group["anchor_id"]

            anchor0 = coords[anchor_id]
            r_rel = coords[group_atoms] - anchor0

            torque_g = np.sum(np.cross(r_rel, G_full[group_atoms]), axis=0)
            G_parts.append(torque_g)

        elif axis_dimension == "2d":
            # dE/dteta = (sum r_i x g_i)·u
            p0, p1 = coords[group["a0"]], coords[group["a1"]]
            axis_vector = p1 - p0
            n = np.linalg.norm(axis_vector)
            u = group["u0"] if n < 1e-12 else axis_vector/n
            r_rel = coords[group_atoms] - p0

            torque_g = np.sum(np.cross(r_rel, G_full[group_atoms]), axis=0)  # (3,)
            dEdteta = np.dot(torque_g, u)
            G_parts.append([dEdteta])
        else:
            assert False, f"Invalid anchor axis dimension"

    G_parts.append(G_full[free_atoms].ravel())
    return np.concatenate(G_parts)