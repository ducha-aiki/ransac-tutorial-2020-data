# select the data
import numpy as np
import h5py
import cv2
from utils import *
from metrics import *
from tqdm import tqdm

def create_cv2_submission(split = 'val', inlier_th = 3.0, match_th = 0.85, n_iter=100000):
    DIR = 'homography'
    out_model = {}
    for ds in ['EVD', 'HPatchesSeq']:
        out_model[ds] = {}
        matches = load_h5(f'{DIR}/{ds}/{split}/matches.h5')
        matches_scores  = load_h5(f'{DIR}/{ds}/{split}/match_conf.h5')
        for k, m in tqdm(matches.items()):
            ms = matches_scores[k].reshape(-1)
            mask = ms <= match_th
            tentatives = m[mask]
            src_pts = tentatives[:,:2]
            dst_pts = tentatives[:,2:]
            H, mask_inl = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 
                                         inlier_th, maxIters=n_iter, confidence=0.9999)
            out_model[ds][k] = H
    return out_model


def evaluate_results(results_dict, split='val'):
    DIR = 'homography'
    MAEs = {}
    for ds in ['EVD', 'HPatchesSeq']:
        Hgt_dict = load_h5(f'{DIR}/{ds}/{split}/Hgt.h5')
        models = results_dict[ds]
        MAEs[ds] = {}
        for k, H_est in tqdm(models.items()):
            H_gt = Hgt_dict[k]
            img1, img2 = get_h_imgpair(k, ds, split)
            MAE = get_visible_part_mean_absolute_reprojection_error(img1, img2, H_gt, H_est)
            MAEs[ds][k] = MAE
    return MAEs


def grid_search_hypers_opencv(INL_THs = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
                             MATCH_THs = [0.75, 0.8, 0.85, 0.9, 0.95]):
    res = {}
    for inl_th in INL_THs:
        for match_th in MATCH_THs:
            key = f'{inl_th}_{match_th}'
            print (f"inlier_th = {inl_th}, snn_ration = {match_th}")
            cv2_results = create_cv2_submission(split = 'val',
                                                inlier_th = inl_th,
                                                match_th = match_th,
                                                n_iter=50000)
            MAEs = evaluate_results(cv2_results)
            mAA = calc_mAA(MAEs)
            final = (mAA['EVD'] + mAA['HPatchesSeq'])/2.0
            print (f'Validation mAA = {final}')
            res[key] = final
    max_MAA = 0
    inl_good = 0
    match_good = 0
    for k, v in res.items():
        if max_MAA < v:
            max_MAA = v
            pars = k.split('_')
            match_good = float(pars[1])
            inl_good =  float(pars[0])
    return inl_good, match_good, max_MAA


if __name__ == '__main__':
    # Search for the best hyperparameters on the validation set
    print ("Searching hypers")
    inl_good, match_good, max_MAA = grid_search_hypers_opencv()
    print (f"The best hyperparameters for OpenCV H RANSAC are")
    print (f"inlier_th = {inl_good}, snn_ration = {match_good}. Validation mAA = {max_MAA}")
    print ("Creating submission")
    cv2_test_submission  = create_cv2_submission(split = 'test', inlier_th = inl_good, match_th = match_good, 
                                             n_iter=50000)
    for ds_name, models in cv2_test_submission.items():
        save_h5(models, f'homography_opencv_{ds_name}_submission.h5')
        print (f"Saved to homography_opencv_{ds_name}_submission.h5")
