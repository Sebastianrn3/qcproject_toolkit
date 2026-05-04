import time
from itertools import product

import numpy as np
from numba import njit, prange


@njit
def even_score_nb(chain, coef, window, eps=1e-12):
    m = chain.shape[0] - 1
    seg = np.empty(m)
    total = 0.0

    for i in range(m):
        seg[i] = np.linalg.norm(chain[i + 1] - chain[i])
        total += seg[i]

    if total < eps:
        return 0.0

    acc = penalty = 0.0
    scale = m / window

    for i in range(m - 1):
        acc += seg[i]
        d = (acc / total - (i + 1) / m) * scale
        penalty += d * d

    return coef * penalty


@njit
def bead_cos_nb(x_prev, x_next, grad, eps=1e-4):
    s = 0.0
    n = 0

    for a in range(x_prev.shape[0]):
        t = x_next[a] - x_prev[a]
        tn = np.linalg.norm(t)
        gn = np.linalg.norm(grad[a])

        if tn > eps and gn > eps:
            s += abs(np.dot(t, grad[a])) / (tn * gn)
            n += 1

    return (s / n if n else 0.0), n


@njit(parallel=True)
def compute_scores_nb(X, G, variants, r_idx, p_idx, coef, window):
    V = variants.shape[0]
    T, A, D = X.shape[0], X.shape[2], X.shape[3]

    scores = np.zeros((V, 3))       # total, cos, P_even
    logs = np.zeros((V, T - 2, 2))  # bead_cos, n_mask

    for v in prange(V):
        combo = variants[v]
        chain = np.empty((T, A, D))

        chain[0] = X[0, r_idx]
        chain[T - 1] = X[T - 1, p_idx]

        for t in range(1, T - 1):
            chain[t] = X[t, combo[t - 1]]

        cos_sum = 0.0

        for t in range(1, T - 1):
            c, n = bead_cos_nb(chain[t - 1], chain[t + 1], G[t, combo[t - 1]])
            cos_sum += c
            logs[v, t - 1, 0] = c
            logs[v, t - 1, 1] = n

        p_even = even_score_nb(chain, coef, window)

        scores[v, 0] = cos_sum - p_even
        scores[v, 1] = cos_sum
        scores[v, 2] = p_even

    return scores, logs


def pack_groups(groups):
    shape = groups[0][0]["geom"].shape
    X = np.zeros((len(groups), max(map(len, groups)), *shape))
    G = np.zeros_like(X)

    for t, group in enumerate(groups):
        for k, item in enumerate(group):
            X[t, k] = item["geom"]
            G[t, k] = item["grad"]

    return X, G


def brute_force_trajectories(groups, coef, window, top_n=50):
    T = len(groups)
    if T < 3:
        raise ValueError("Need at least 3 groups: R, inner beads, P.")

    inner_sizes = [len(g) for g in groups[1:-1]]
    if any(k == 0 for k in inner_sizes):
        raise ValueError(f"At least one inner group has zero candidates: {inner_sizes}")

    r_idx = len(groups[0]) - 1
    p_idx = len(groups[-1]) - 1

    n_combos = int(np.prod(inner_sizes))
    print(f"Images: {T} | Fixed: R,P | Inner sizes: {inner_sizes} | Combos: {n_combos}")

    X, G = pack_groups(groups)

    t0 = time.time()
    variants = np.array(list(product(*[range(k) for k in inner_sizes])), dtype=np.int32)
    print(f"Matrix ready in {time.time() - t0:.2f}s")

    t0 = time.time()
    scores, logs = compute_scores_nb(X, G, variants, r_idx, p_idx, coef, window)
    print(f"Computation done in {time.time() - t0:.2f}s")

    best = np.argsort(scores[:, 0])[::-1][:min(top_n, len(scores))]

    cos = logs[..., 0].ravel()
    nmask = logs[..., 1].ravel()

    print(
        f"\nDiagnostics:\n"
        f"  cos mean/p95       = {cos.mean():.4g} / {np.percentile(cos, 95):.4g}\n"
        f"  cos min/max        = {cos.min():.4g} / {cos.max():.4g}\n"
        f"  n_mask min/max/avg = {int(nmask.min())} / {int(nmask.max())} / {nmask.mean():.2f}\n"
        f"  P_even window/coef = {window} / {coef}"
    )

    out = []

    for rank, i in enumerate(best, 1):
        inner = tuple(map(int, variants[i]))
        full = (int(r_idx), *inner, int(p_idx))

        item = {
            "total_score": float(scores[i, 0]),
            "cos_score": float(scores[i, 1]),
            "even_score": float(scores[i, 2]),
            "inner_combo": inner,
            "full_combo": full,
            "bead_cos": logs[i, :, 0].copy(),
            "bead_n_mask": logs[i, :, 1].astype(int).copy(),
        }

        out.append(item)

        report_ranks = {1, 2, 5, 25, 100, 300}

        if rank in report_ranks:
            print(
                f"{rank}. total = {item['total_score']:.6f}, "
                f"cos = {item['cos_score']:.6f}, "
                f"P_even = {item['even_score']:.6f}\n"
                f"   inner combo = {inner}\n"
                f"   full combo  = {full}\n"
                f"   bead cos    = {np.array2string(item['bead_cos'], precision=4)}\n"
                f"   bead n_mask = {item['bead_n_mask']}"
            )

    return out