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
from skimage.transform import ProjectiveTransform
import multiprocessing
import sys
from joblib import Parallel, delayed
import PIL


def get_single_result(ms, m, method, params):
    mask = (ms <= params['match_th']).reshape(-1)
    tentatives = m[mask]
    tentative_idxs = np.arange(len(mask))[mask]
    src_pts = tentatives[:, :2]
    dst_pts = tentatives[:, 2:]
    if tentatives.shape[0] <= 5:
        return np.eye(3), np.array([False] * len(mask))
    if method == 'cv2h':
        H, mask_inl = cv2.findHomography(src_pts, dst_pts, 
                                                cv2.RANSAC, 
                                                params['inl_th'],
                                                maxIters=params['maxiter'],
                                                confidence=params['conf'])
    elif method  == 'pyransac':
        H, mask_inl = pydegensac.findHomography(src_pts, dst_pts, 
                                                params['inl_th'],
                                                conf=params['conf'],
                                                max_iters = params['maxiter'])
    elif method  == 'sklearn':
        try:
            #print(src_pts.shape, dst_pts.shape)
            H, mask_inl = skransac([src_pts, dst_pts],
                        ProjectiveTransform,
                        min_samples=4,
                        residual_threshold=params['inl_th'],
                        max_trials=params['maxiter'],
                        stop_probability=params['conf'])
            mask_inl = mask_inl.astype(bool).flatten()
            H = H.params
        except Exception as e:
            print ("Fail!", e)
            return np.eye(3), np.array([False] * len(mask))
    else:
        raise ValueError('Unknown method')
    
    final_inliers = np.array([False] * len(mask))
    if H is not None:
        for i, x in enumerate(mask_inl):
            final_inliers[tentative_idxs[i]] = x
    return H, final_inliers


def create_H_submission(IN_DIR, seq,  method, params = {}):
    out_model = {}
    inls = {}
    matches = load_h5(f'{IN_DIR}/matches.h5')
    matches_scores = load_h5(f'{IN_DIR}/match_conf.h5')
    keys = [k for k in matches.keys()]
    results = Parallel(n_jobs=min(num_cores,len(keys)))(delayed(get_single_result)(matches_scores[k], matches[k], method, params) for k in tqdm(keys))
    for i, k in enumerate(keys):
        v = results[i]
        out_model[k] = v[0]
        inls[k] = v[1]
    return  out_model, inls



if __name__ == '__main__':
    supported_methods = [ 'cv2h', 'pyransac', 'sklearn', 'cv2lmeds']
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default='val',
        type=str,
        help='split to run on. Can be val or test')
    parser.add_argument(
        "--method", default='cv2h', type=str,
        help=f'can be {supported_methods}' )
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
        default='h_data',
        type=str,
        help='path to the data')
    
    args = parser.parse_args()

    if args.split not in ['val', 'test']:
        raise ValueError('Unknown value for --split')
    
    if args.method.lower() not in supported_methods:
        raise ValueError(f'Unknown value {args.method.lower()} for --method')
    NUM_RUNS = 1
    if args.split == 'test':
        NUM_RUNS = 3
    params = {"maxiter": args.maxiter,
              "inl_th": args.inlier_th,
              "conf": args.conf,
              "match_th": args.match_th
    }
    problem = 'h'
    OUT_DIR = get_output_dir(problem, args.split, args.method, params)
    IN_DIR = args.data_dir
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    num_cores = int(len(os.sched_getaffinity(0)) * 0.5)
    for run in range(NUM_RUNS):
        seqs = os.listdir(IN_DIR)
        for seq in seqs:
            IN_DIR_CURRENT = os.path.join(IN_DIR, seq, args.split)
            print (f'Working on {seq}')
            out_models_fname = os.path.join(OUT_DIR, f'submission_models_seq_{seq}_run_{run}.h5')
            out_inliers_fname = os.path.join(OUT_DIR, f'submission_inliers_seq_{seq}_run_{run}.h5')
            
            if os.path.isfile(out_models_fname) and not args.force:
                print (f"Submission file {out_models_fname} already exists, skipping")
                continue
            models, inlier_masks = create_H_submission(IN_DIR_CURRENT, seq,
                                    args.method,
                                    params)
            save_h5(models, out_models_fname)
            save_h5(inlier_masks, out_inliers_fname)
    print ('Done!')
