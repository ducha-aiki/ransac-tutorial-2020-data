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
import multiprocessing
import sys
from joblib import Parallel, delayed
import PIL


def eval_single_result(k, H_gt, H, IN_DIR):
    img1, img2 = get_h_imgpair2(k, IN_DIR)
    return get_visible_part_mean_absolute_reprojection_error(img1, img2, H_gt, H)

def evaluate_results(IN_DIR, seq,  models, inliers):
    MAEs = {}
    Hgt_dict = load_h5(f'{IN_DIR}/Hgt.h5')
    H_pred, inl_mask = models, inliers
    keys = sorted([k for k in H_pred.keys()])
    num_cores = int(len(os.sched_getaffinity(0)) * 0.9)
    results = Parallel(n_jobs=num_cores)(delayed(eval_single_result)(k, Hgt_dict[k], H_pred[k], IN_DIR) for k in tqdm(keys))
    for i, k in enumerate(keys):
        MAEs[k] = v = results[i]
    return MAEs



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
    all_maas = []
    for run in range(NUM_RUNS):
        seqs = os.listdir(IN_DIR)
        for seq in seqs:
            IN_DIR_CURRENT = os.path.join(IN_DIR, seq, args.split)
            print (f'Working on {seq}')
            in_models_fname = os.path.join(OUT_DIR, f'submission_models_seq_{seq}_run_{run}.h5')
            in_inliers_fname = os.path.join(OUT_DIR, f'submission_inliers_seq_{seq}_run_{run}.h5')
            out_errors_fname = os.path.join(OUT_DIR, f'errors_seq_{seq}_run_{run}.h5')
            out_maa_fname = os.path.join(OUT_DIR, f'maa_seq_{seq}_run_{run}.h5')
            if not os.path.isfile(in_models_fname) or not os.path.isfile(in_inliers_fname):
                print (f"Submission file {in_inliers_fname} is missing, cannot evaluate, skipping")
                continue
            models = load_h5(in_models_fname)
            inlier_masks = load_h5(in_inliers_fname)
            if os.path.isfile(out_errors_fname) and not args.force:
                print (f"Submission file {in_inliers_fname} exists, read it")
                error = load_h5(out_errors_fname)
            else:
                error = evaluate_results(IN_DIR_CURRENT, seq,  models, inlier_masks)
            save_h5(error, out_errors_fname)
            mAA = calc_mAA({seq: error})
            print (f" mAA {seq} = {mAA[seq]:.5f}")
            save_h5({"mAA": mAA[seq]}, out_maa_fname)
            all_maas.append(mAA[seq])
    out_maa_final_fname = os.path.join(OUT_DIR, f'maa_FINAL.h5')
    final_mAA = (np.array(all_maas)).mean()
    print (f" mAA total = {final_mAA:.5f}")
    save_h5({"mAA": final_mAA}, out_maa_final_fname)
    print ('Done!')
        


