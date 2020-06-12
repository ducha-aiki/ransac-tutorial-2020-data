import cv2
import numpy as np


def get_visible_part_mean_absolute_reprojection_error(img1, img2, H_gt, H):
    '''We reproject the image 1 mask to image2 and back to get the visible part mask.
    Then we average the reprojection absolute error over that area'''
    h,w = img1.shape[:2]
    mask1 = np.ones((h,w))
    mask1in2 = cv2.warpPerspective(mask1, H_gt, img2.shape[:2][::-1])
    mask1inback = cv2.warpPerspective(mask1in2, np.linalg.inv(H_gt), img1.shape[:2][::-1]) > 0
    xi = np.arange(w)
    yi = np.arange(h)
    xg, yg = np.meshgrid(xi,yi)
    coords = np.concatenate([xg.reshape(*xg.shape,1), yg.reshape(*yg.shape,1)], axis=-1)
    shape_orig = coords.shape
    xy_rep_gt = cv2.perspectiveTransform(coords.reshape(-1, 1,2).astype(np.float32), H_gt.astype(np.float32)).squeeze(1)
    xy_rep_estimated = cv2.perspectiveTransform(coords.reshape(-1, 1,2).astype(np.float32),
                                                H.astype(np.float32)).squeeze(1)
    #error = np.abs(xy_rep_gt-xy_rep_estimated).sum(axis=1).reshape(xg.shape) * mask1inback
    error = np.sqrt(((xy_rep_gt-xy_rep_estimated)**2).sum(axis=1)).reshape(xg.shape) * mask1inback
    mean_error = error.sum() / mask1inback.sum()
    return mean_error


def calc_mAA(MAEs, ths = np.logspace(np.log2(1.0), np.log2(20), 10, base=2.0)):
    res = {}
    for ds_name, MAEs_cur in MAEs.items():
        cur_results = []
        for k, MAE in MAEs_cur.items():
            acc = []
            for th in ths:
                A = (MAE <= th).astype(np.float32).mean()
                acc.append(A)
            cur_results.append(np.array(acc).mean())
        res[ds_name] = np.array(cur_results).mean()
    return res

def calc_mAA_FE(ang_errors, ths = np.deg2rad(np.linspace(1.0, 10., 10))):
    res = {}
    for ds_name, MAEs_cur in ang_errors.items():
        cur_results = []
        for k, MAE in MAEs_cur.items():
            acc = []
            for th in ths:
                A = (MAE <= th).astype(np.float32).mean()
                acc.append(A)
            cur_results.append(np.array(acc).mean())
        res[ds_name] = np.array(cur_results).mean()
    return res
