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
from skimage.transform import EssentialMatrixTransform
import multiprocessing

from joblib import Parallel, delayed
import sys




def get_single_result(ms, m, method, K1, K2, params):
    if args.method.lower() == 'load_oanet': #They provided not the mask, but actual correspondences
        tentatives = ms
    else:
        final_inliers = np.array([False] * len(mask))
        mask = ms
        tentatives = m[mask]
        tentative_idxs = np.arange(len(mask))[mask]
    src_pts = normalize_keypoints(tentatives[:, :2], K1)
    dst_pts = normalize_keypoints(tentatives[:, 2:], K2)
    
    if tentatives.shape[0] <= 10:
        return np.eye(3), np.array([False] * len(mask))
    E, mask_inl = cv2.findEssentialMat(src_pts, dst_pts, 
                                        np.eye(3), cv2.RANSAC, 
                                        threshold=0.0002,
                                        prob=0.999)
    mask_inl = mask_inl.astype(bool).flatten()
    if args.method.lower() == 'load_oanet': #They provided not the mask, but actual correspondences
        final_inliers = tentatives[mask_inl]
    else:
        if E is not None:
            for i, x in enumerate(mask_inl):
                final_inliers[tentative_idxs[i]] = x
    return E, final_inliers


def upgrade_E_submission(IN_DIR, inliers, seq,  method, params = {}):
    out_model = {}
    inls = {}
    matches = load_h5(f'{IN_DIR}/{seq}/matches.h5')
    #matches_scores = load_h5(f'{IN_DIR}/{seq}/match_conf.h5')
    K1_K2 = load_h5(f'{IN_DIR}/{seq}/K1_K2.h5')
    keys = [k for k in matches.keys()]

    results = Parallel(n_jobs=num_cores)(delayed(get_single_result)(inliers[k], matches[k], method, K1_K2[k][0][0], K1_K2[k][0][1], params) for k in tqdm(keys))
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default='val',
        type=str,
        help='split to run on. Can be val or test')
    parser.add_argument(
        "--method", default='cv2e', type=str,
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
    
    if args.method.lower() not in ['cne', 'acne','load_dfe', 'cv2e', 'sklearn', 'nmnet2', 'oanet2', 'load_oanet', 'load_oanet_ransac']:
        raise ValueError('Unknown value for --method')
    NUM_RUNS = 1
    if args.split == 'test':
        NUM_RUNS = 1
    params = {"maxiter": args.maxiter,
              "inl_th": args.inlier_th,
              "conf": args.conf,
              "match_th": args.match_th
    }
    problem = 'e'
    OUT_DIR = get_output_dir(problem, args.split, args.method, params)
    IN_DIR = os.path.join(args.data_dir, args.split) 
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    num_cores = int(len(os.sched_getaffinity(0)) * 0.9)
    for run in range(NUM_RUNS):
        seqs = os.listdir(IN_DIR)
        for seq in seqs:
            print (f'Working on {seq}')
            in_models_fname = os.path.join(OUT_DIR, f'submission_models_seq_{seq}_run_{run}.h5')
            in_inliers_fname = os.path.join(OUT_DIR, f'submission_inliers_seq_{seq}_run_{run}.h5')
            if args.method.lower() == 'load_dfe':
                in_inliers_fname = in_inliers_fname.replace('/e/','/f/')
            out_models_fname = os.path.join(OUT_DIR, f'submission_upgraded_models_seq_{seq}_run_{run}.h5')
            out_inliers_fname = os.path.join(OUT_DIR, f'submission_upgraded_inliers_seq_{seq}_run_{run}.h5')
            if os.path.isfile(out_models_fname) and not args.force:
                print (f"Submission file {out_models_fname} already exists, skipping")
                continue
            inliers = load_h5(in_inliers_fname)
            models, inlier_masks = upgrade_E_submission(IN_DIR, inliers, seq,
                                    args.method,
                                    params)
            save_h5(models, out_models_fname)
            save_h5(inlier_masks, out_inliers_fname)
    print ('Done!')
        
        


