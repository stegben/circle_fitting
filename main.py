import argparse

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tqdm
from scipy import linalg

from ransac import fit_circle_ransac


FILE_NAME = '97909.csv'
GROUND_TRUTH_PATH = 'centers.npy'
MAX_COBRA_ID = 2394


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiver_plot_path", type=str)
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--algo", type=str, choices=['analytic', 'ransac'])
    return parser.parse_args()


def remove_outliers_by_median(data_points, margin=1.5):
    nd = np.abs(data_points - np.median(data_points))
    s = nd/np.median(nd)
    return data_points[s<margin]


def find_center_analytic(points_arr):
    arr_mean = np.mean(points_arr, axis=0)
    diff = points_arr - arr_mean
    u = diff[:, 0]
    v = diff[:, 1]
    uv = u * v
    uu = u ** 2
    vv = v ** 2
    uuv = uu * v
    uvv = vv * u
    uuu = u * uu
    vvv = v * vv

    A = np.array([[uu.sum(), uv.sum()], [uv.sum(), vv.sum()]])
    B = np.array([uuu.sum() + uvv.sum(), vvv.sum() + uuv.sum()]) / 2.0
    uc, vc = linalg.solve(A, B)

    return arr_mean + np.array([uc, vc])


def find_center_ransac(points_arr):
    cx, cy, r, _ = fit_circle_ransac(
        points_arr[:, 0],
        points_arr[:, 1],
        num_iterations=10,
        threshold=0.05,
    )
    return np.array([cx, cy])


def read_ground_truth(path):
    arr = np.load(path)
    new_arr = np.zeros(shape=(arr.shape[0], 2), dtype=np.float64)
    new_arr[:, 0] = arr.real
    new_arr[:, 1] = arr.imag
    # import ipdb; ipdb.set_trace()
    assert new_arr.shape[0] == MAX_COBRA_ID
    return new_arr


def l2_diff(estimated, ground_truth, exclude_idxs=None):
    if exclude_idxs is None:
        exclude_idxs = []
    mask = np.ones(estimated.shape[0]).astype(bool)
    for exclude_idx in exclude_idxs:
        mask[exclude_idx] = False
    l2 = ((estimated - ground_truth) ** 2).sum(axis=1)
    # import ipdb; ipdb.set_trace()
    return np.sqrt(l2.mean(where=mask))


def main():
    args = parse_args()
    df = pd.read_csv(args.input_path)
    df.cobra_id -= 1
    # loop through all cobra
    all_cobra = set(df.cobra_id.unique())
    exclude_idxs = set(i for i in range(MAX_COBRA_ID)) - all_cobra
    arr_estimated = np.zeros(shape=(MAX_COBRA_ID, 2), dtype=np.float64)
    arr_ground_truth = read_ground_truth(GROUND_TRUTH_PATH)
    
    for cobra_id in tqdm.tqdm(all_cobra):
        if args.algo == 'analytic':
            arr_estimated[cobra_id] = find_center_analytic(
                df[df['cobra_id'] == cobra_id][['pfi_center_x_mm', 'pfi_center_y_mm']].to_numpy())
        elif args.algo == 'ransac': 
            arr_estimated[cobra_id] = find_center_ransac(
                df[df['cobra_id'] == cobra_id][['pfi_center_x_mm', 'pfi_center_y_mm']].to_numpy())
    for cobra_id in exclude_idxs:
        arr_estimated[cobra_id] = arr_ground_truth[cobra_id, :]
    
    diff = arr_estimated - arr_ground_truth
    plt.quiver(arr_ground_truth[:, 0], arr_ground_truth[:, 1], diff[:, 0], diff[:, 1], angles='uv')
    plt.savefig(args.quiver_plot_path)

    print(l2_diff(arr_estimated, arr_ground_truth, exclude_idxs))
    l2 = ((arr_estimated - arr_ground_truth) ** 2).sum(axis=1)
    import ipdb; ipdb.set_trace()

main()
# data= df[df['cobra_id'] == 2]
# fig, ax = plt.subplots(figsize=(8,8), facecolor="white")
# ax.plot(data['pfi_center_x_mm'],data['pfi_center_y_mm'],'.')
# plt.savefig('cobra2.png')
