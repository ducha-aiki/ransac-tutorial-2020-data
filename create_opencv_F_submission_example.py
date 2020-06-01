# select the data
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
from utils import *
from tqdm import tqdm
import os
from metrics import *



def create_F_submission_cv2(split = 'val', inlier_th = 1.0, match_th = 0.8):
    DIR = split
    seqs = os.listdir(DIR)
    out_model = {}
    inls = {}
    for seq in seqs:
        matches = load_h5(f'{DIR}/{seq}/matches.h5')
        matches_scores = load_h5(f'{DIR}/{seq}/match_conf.h5')
        out_model[seq] = {}
        inls[seq] = {}
        for k, m in tqdm(matches.items()):
            ms = matches_scores[k].reshape(-1)
            mask = ms <= match_th
            tentatives = m[mask]
            src_pts = tentatives[:,:2]
            dst_pts = tentatives[:,2:]
            F, mask_inl = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 
                                         inlier_th, confidence=0.9999)
            out_model[seq][k] = F
            inls[seq][k] = mask_inl
    return  out_model, inls

def evaluate_results(submission, split = 'val'):
    ang_errors = {}
    DIR = split
    seqs = os.listdir(DIR)
    for seq in seqs:
        matches = load_h5(f'{DIR}/{seq}/matches.h5')
        K1_K2 = load_h5(f'{DIR}/{seq}/K1_K2.h5')
        R = load_h5(f'{DIR}/{seq}/R.h5')
        T = load_h5(f'{DIR}/{seq}/T.h5')
        F_pred, inl_mask = submission[0][seq], submission[1][seq]
        ang_errors[seq] = {}
        for k, m in tqdm(matches.items()):
            if F_pred[k] is None:
                ang_errors[seq][k] = 3.14
                continue
            img_id1 = k.split('-')[0]
            img_id2 = k.split('-')[1]
            K1 = K1_K2[k][0][0]
            K2 = K1_K2[k][0][1]
            try:
                E_cv_from_F = get_E_from_F(F_pred[k], K1, K2)
            except:
                print ("Fail")
                E = np.eye(3)
            R1 = R[img_id1]
            R2 = R[img_id2]
            T1 = T[img_id1]
            T2 = T[img_id2]
            dR = np.dot(R2, R1.T)
            dT = T2 - np.dot(dR, T1)
            pts1 = m[inl_mask[k],:2] # coordinates in image 1
            pts2 = m[inl_mask[k],2:]  # coordinates in image 2
            p1n = normalize_keypoints(pts1, K1)
            p2n = normalize_keypoints(pts2, K2)
            ang_errors[seq][k] = max(eval_essential_matrix(p1n, p2n, E_cv_from_F, dR, dT))
    return ang_errors


def grid_search_hypers_opencv(INL_THs = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0],
                             MATCH_THs = [0.75, 0.8, 0.85, 0.9, 0.95]):
    res = {}
    for inl_th in INL_THs:
        for match_th in MATCH_THs:
            key = f'{inl_th}_{match_th}'
            print (f"inlier_th = {inl_th}, snn_ration = {match_th}")
            cv2_results = create_F_submission_cv2(split = 'val',
                                                inlier_th = inl_th,
                                                match_th = match_th)
            MAEs = evaluate_results(cv2_results, 'val')
            mAA = calc_mAA_FE(MAEs)
            final = 0
            for k,v in mAA.items():
                final+= v / float(len(mAA))
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
    cv2_test_submission  = create_F_submission_cv2(split = 'test', inlier_th = inl_good, match_th = match_good)
    for ds_name, models in cv2_test_submission.items():
        save_h5(models, f'F_opencv_{ds_name}_submission.h5')
        print (f"Saved to F_opencv_{ds_name}_submission.h5")
        


