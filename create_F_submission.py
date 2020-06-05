# select the data
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
from utils import *
from tqdm import tqdm
import os
from metrics import *
import argparse
import pydegensac

from skimage.measure import ransac as skransac
from skimage.transform import FundamentalMatrixTransform
import multiprocessing

from joblib import Parallel, delayed

def get_single_result(ms, m, method, params):
    mask = ms <= params['match_th']
    tentatives = m[mask]
    tentative_idxs = np.arange(len(mask))[mask]
    src_pts = tentatives[:,:2]
    dst_pts = tentatives[:,2:]
    if len(dst_pts) <= 8:
        return np.eye(3), np.array([False] * len(mask))
    if method == 'cv2f':
        F, mask_inl = cv2.findFundamentalMat(src_pts, dst_pts, 
                                                cv2.RANSAC, 
                                                params['inl_th'],
                                                confidence=params['conf'])
    elif method  == 'pyransac':
        F, mask_inl = pydegensac.findFundamentalMatrix(src_pts, dst_pts, 
                                                params['inl_th'],
                                                conf=params['conf'],
                                                max_iters = params['maxiter'],
                                                enable_degeneracy_check=False)
    elif method  == 'degensac':
        F, mask_inl = pydegensac.findFundamentalMatrix(src_pts, dst_pts, 
                                                params['inl_th'],
                                                conf=params['conf'],
                                                max_iters = params['maxiter'],
                                                enable_degeneracy_check=True)
    elif method  == 'sklearn':
        F, mask_inl = skransac((src_pts, dst_pts),
                        FundamentalMatrixTransform,
                        min_samples=8,
                        residual_threshold=params['inl_th'],
                        max_trials=params['maxiter'],
                        stop_probability=params['conf'])
    else:
        raise ValueError('Unknown method')
    
    final_inliers = np.array([False] * len(mask))
    if F is not None:
        for i, x in enumerate(mask_inl):
            final_inliers[tentative_idxs[i]] = x
    return F, final_inliers

        
def create_F_submission(IN_DIR,seq,  method, params = {}):
    out_model = {}
    inls = {}
    matches = load_h5(f'{IN_DIR}/{seq}/matches.h5')
    matches_scores = load_h5(f'{IN_DIR}/{seq}/match_conf.h5')
    keys = [k for k in matches.keys()]
    results = Parallel(n_jobs=num_cores)(delayed(get_single_result)(matches_scores[k], matches[k], method, params) for k in tqdm(keys))
    for i, k in enumerate(keys):
        v = results[i]
        out_model[k] = v[0]
        inls[k] = v[1]
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default='val',
        type=str,
        help='split to run on. Can be val or test')
    parser.add_argument(
        "--method", default='cv2F', type=str,
        help=' can be cv2f, pyransac, degensac, sklearn' )
    parser.add_argument(
        "--inlier_th",
        default=0.75,
        type=float,
        help='inlier threshold. Default is 0.75')
    parser.add_argument(
        "--conf",
        default=0.999,
        type=float,
        help='confidence Default is 0.999')
    parser.add_argument(
        "--maxiter",
        default=100000,
        type=int,
        help='max iter Default is 100000')
    parser.add_argument(
        "--match_th",
        default=0.85,
        type=float,
        help='match filetring th. Default is 0.85')
    
    parser.add_argument(
        "--force",
        default=False,
        type=bool,
        help='Force recompute if exists')
    parser.add_argument(
        "--data_dir",
        default='f_data',
        type=str,
        help='path to the data')
    
    args = parser.parse_args()

    if args.split not in ['val', 'test']:
        raise ValueError('Unknown value for --split')
    
    if args.method.lower() not in ['cv2f', 'pyransac', 'degensac', 'sklearn']:
        raise ValueError('Unknown value for --method')
    NUM_RUNS = 1
    if args.split == 'test':
        NUM_RUNS = 3
    params = {"maxiter": args.maxiter,
              "inl_th": args.inlier_th,
              "conf": args.conf,
              "match_th": args.match_th
    }
    problem = 'f'
    OUT_DIR = get_output_dir(problem, args.split, args.method, params)
    IN_DIR = os.path.join(args.data_dir, args.split) 
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    num_cores = int(len(os.sched_getaffinity(0)) * 0.9)
    for run in range(NUM_RUNS):
        seqs = os.listdir(IN_DIR)
        for seq in seqs:
            print (f'Working on {seq}')
            out_models_fname = os.path.join(OUT_DIR, f'submission_models_seq_{seq}_run_{run}.h5')
            out_inliers_fname = os.path.join(OUT_DIR, f'submission_inliers_seq_{seq}_run_{run}.h5')
            
            if os.path.isfile(out_models_fname) and not args.force:
                print (f"Submission file {out_models_fname} already exists, skipping")
                continue
            models, inlier_masks = create_F_submission(IN_DIR, seq,
                                    args.method,
                                    params)
            save_h5(models, out_models_fname)
            save_h5(inlier_masks, out_inliers_fname)
    print ('Done!')
        
        


