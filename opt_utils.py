# -*- coding: utf-8 -*-
"""
Created on Sun May 18 11:18:13 2025

@author: Yoyooooo
"""

import jax.numpy as jnp
import numpy as np
from scipy.sparse import coo_matrix
from itertools import product
from collections import defaultdict


def var_index(i, j):
    return 6 * i + j


def get_seg_indices(seg):
    seg_map = defaultdict(list)
    for idx, s in enumerate(seg):
        seg_map[int(s)].append(idx)
    return {k: np.array(v) for k, v in seg_map.items()}


def target_constraint_sparse(N):
    row_idx = np.repeat(np.arange(N), 6)
    col_idx = np.array([var_index(i, j) for i in range(N) for j in range(6)])
    data = -np.ones_like(row_idx)
    G = coo_matrix((data, (row_idx, col_idx)), shape=(N, 6 * N))
    h = -np.ones((N,))
    return G, h


def volume_constraint_I_sparse(N, seg, a, b):
    seg_indices = get_seg_indices(seg)
    rows, cols, data, rhs = [], [], [], []
    row_count = 0

    for j in range(5):
        for sg, indices in seg_indices.items():
            n_k = len(indices)
            for i in indices:
                rows.append(row_count)
                cols.append(var_index(i, j))
                data.append(1.0)
            rhs.append(a[j] * n_k)
            row_count += 1

            for i in indices:
                rows.append(row_count)
                cols.append(var_index(i, j))
                data.append(-1.0)
            rhs.append(-b[j] * n_k)
            row_count += 1

    G = coo_matrix((data, (rows, cols)), shape=(row_count, 6 * N))
    h = np.array(rhs)
    return G, h


def volume_constraint_II_sparse(N, seg):
    seg_indices = get_seg_indices(seg)
    rows, cols, data, rhs = [], [], [], []
    row_count = 0

    for sg, indices in seg_indices.items():
        for i in indices:
            for j in range(5):
                rows.append(row_count)
                cols.append(var_index(i, j))
                data.append(1.0)
        rhs.append(0.7 * len(indices))
        row_count += 1

    G = coo_matrix((data, (rows, cols)), shape=(row_count, 6 * N))
    h = np.array(rhs)
    return G, h


def similarity_constraint_I_sparse(N, seg, lmbdas):
    seg_indices = get_seg_indices(seg)
    seg_keys = list(seg_indices.keys())
    K = len(seg_keys)

    rows, cols, data, rhs = [], [], [], []
    row_count = 0

    for j in range(6):
        for k1, k2 in product(range(K), repeat=2):
            if k1 == k2:
                continue
            sg1, sg2 = seg_keys[k1], seg_keys[k2]
            idx1, idx2 = seg_indices[sg1], seg_indices[sg2]
            n1, n2 = len(idx1), len(idx2)
            lam = lmbdas[j][k1][k2]

            for i in idx1:
                rows.append(row_count)
                cols.append(var_index(i, j))
                data.append(lam / n1)

            for i in idx2:
                rows.append(row_count)
                cols.append(var_index(i, j))
                data.append(-1.0 / n2)

            rhs.append(0.0)
            row_count += 1

    G = coo_matrix((data, (rows, cols)), shape=(row_count, 6 * N))
    h = np.array(rhs)
    return G, h


def similarity_constraint_II_sparse(N, seg, di, gammas):
    seg_indices = get_seg_indices(seg)
    seg_keys = list(seg_indices.keys())
    K = len(seg_keys)

    rows, cols, data, rhs = [], [], [], []
    row_count = 0

    for k1, k2 in product(range(K), repeat=2):
        if k1 == k2:
            continue
        sg1, sg2 = seg_keys[k1], seg_keys[k2]
        idx1, idx2 = seg_indices[sg1], seg_indices[sg2]
        n1, n2 = len(idx1), len(idx2)
        gamma = gammas[k1][k2]

        for i in idx1:
            for j in range(5):
                rows.append(row_count)
                cols.append(var_index(i, j))
                data.append(gamma * di[j] / n1)

        for i in idx2:
            for j in range(5):
                rows.append(row_count)
                cols.append(var_index(i, j))
                data.append(-di[j] / n2)

        rhs.append(0.0)
        row_count += 1

    G = coo_matrix((data, (rows, cols)), shape=(row_count, 6 * N))
    h = np.array(rhs)
    return G, h


def build_G_and_h_sparse(N, seg, a, b, di, lmbdas, gammas):
    G_list = []
    h_list = []

    funcs = [
        target_constraint_sparse,
        lambda N: volume_constraint_I_sparse(N, seg, a, b),
        lambda N: volume_constraint_II_sparse(N, seg),
        lambda N: similarity_constraint_I_sparse(N, seg, lmbdas),
        lambda N: similarity_constraint_II_sparse(N, seg, di, gammas),
    ]

    for f in funcs:
        G, h = f(N)
        G_list.append(G)
        h_list.append(h)

    G_final = coo_matrix(np.vstack([G.toarray() for G in G_list]))
    h_final = np.concatenate(h_list)
    return G_final, h_final
