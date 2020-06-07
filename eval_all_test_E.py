import numpy as np
import h5py
import cv2
from utils import *
from tqdm import tqdm
import os
from metrics import *

import argparse


if __name__ == '__main__':
    # Search for the best hyperparameters on the validation set
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default='f_data',
        type=str,
        help='path to the data')
    args = parser.parse_args()
    BASE = f'results/test/e/'
    res = {}
    for method in os.listdir(BASE):
        B1 =  f'{BASE}/{method}/'
        for hypers in  os.listdir(B1):
            key = f'{method}-{hypers}'
            print (f"Evaluating {key}")
            parts = hypers.split('_')
            conf = float(parts[1].split('-')[-1])
            inl_th = float(parts[3].split('-')[-1]) 
            match_th = float(parts[5].split('-')[-1]) 
            maxiters = int(parts[6].split('-')[-1])
            run_str_eval = f'python eval_E_submission.py --data_dir {args.data_dir} --conf {conf} --match_th {match_th} --inlier_th {inl_th} --method {method} --split test --maxiter {maxiters}'
            os.system(run_str_eval)
            params =   {"maxiter": maxiters,
              "inl_th": inl_th,
              "conf": conf,
              "match_th": match_th
            }
            OUT_DIR = get_output_dir('e', 'test', method, params)
            out_maa_final_fname = os.path.join(OUT_DIR, f'maa_FINAL.h5')
            final_res = load_h5(out_maa_final_fname)
            res[key] = final_res['mAA']
            print (final_res['mAA']) 
