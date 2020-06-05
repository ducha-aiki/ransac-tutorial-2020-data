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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", default='cv2F', type=str,
        help=' can be cv2f, pyransac, degensac, sklearn' )
    parser.add_argument(
        "--data_dir",
        default='f_data',
        type=str,
        help='path to the data')
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
    
    args = parser.parse_args()

    print (f"Searching hypers for {args.method}, conf={args.conf}, maxIters={args.maxiter}")
    inl_ths = [0.25, 0.5, 0.75]#, 1.0, 1.5, 2.0]
    match_ths = [0.8, 0.85]
    res = {}
    for m_th in match_ths:
        for inl_th in inl_ths:
            key = f'{inl_th}_{m_th}'
            print (f'inlier threshold = {inl_th}, match threshold={m_th}')
            run_str = f'python create_F_submission.py --data_dir {args.data_dir} --conf {args.conf} --match_th {m_th} --inlier_th {inl_th} --method {args.method} --split val --maxiter {args.maxiter}'
            os.system(run_str)
            run_str_eval = f'python eval_F_submission.py --data_dir {args.data_dir} --conf {args.conf} --match_th {m_th} --inlier_th {inl_th} --method {args.method} --split val --maxiter {args.maxiter}'
            os.system(run_str_eval)
            params =   {"maxiter": args.maxiter,
              "inl_th": inl_th,
              "conf": args.conf,
              "match_th": m_th
            }
            OUT_DIR = get_output_dir('f', 'val', args.method, params)
            out_maa_final_fname = os.path.join(OUT_DIR, f'maa_FINAL.h5')
            final_res = load_h5(out_maa_final_fname)
            res[key] = final_res['mAA']
    max_MAA = 0
    inl_good = 0
    match_good = 0
    for k, v in res.items():
        if max_MAA < v:
            max_MAA = v
            pars = k.split('_')
            match_good = float(pars[1])
            inl_good =  float(pars[0])
    print (f"The best hyperparameters  for {args.method}, conf={args.conf}, maxIters={args.maxiter} are")
    print (f"inlier_th = {inl_good}, snn_ration = {match_good}. Validation mAA = {max_MAA}")
    print ("Creating submission")
    run_str = f'python create_F_submission.py --data_dir {args.data_dir} --conf {args.conf} --match_th {match_good} --inlier_th {inl_good} --method {args.method} --split test --maxiter {args.maxiter}'
    os.system(run_str)
    print ('Done!')
