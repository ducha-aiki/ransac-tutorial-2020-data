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



        

def evaluate_results(IN_DIR, seq,  models, inliers):
    ang_errors = {}
    matches = load_h5(f'{IN_DIR}/{seq}/matches.h5')
    K1_K2 = load_h5(f'{IN_DIR}/{seq}/K1_K2.h5')
    R = load_h5(f'{IN_DIR}/{seq}/R.h5')
    T = load_h5(f'{IN_DIR}/{seq}/T.h5')
    E_pred, inl_mask = models, inliers
    for k, m in tqdm(matches.items()):
        if E_pred[k] is None:
            ang_errors[k] = 3.14
            continue
        img_id1 = k.split('-')[0]
        img_id2 = k.split('-')[1]
        K1 = K1_K2[k][0][0]
        K2 = K1_K2[k][0][1]
        E = E_pred[k].astype(np.float64)
        R1 = R[img_id1]
        R2 = R[img_id2]
        T1 = T[img_id1]
        T2 = T[img_id2]
        dR = np.dot(R2, R1.T)
        dT = T2 - np.dot(dR, T1)
        if args.method.lower() == 'load_oanet': #They provided not the mask, but actual correspondences
            pts1 = inl_mask[k][:, :2] # coordinates in image 1
            pts2 = inl_mask[k][:, 2:]  # coordinates in image 2
        elif args.method.lower() == 'load_oanet_ransac': #They provided not the mask, but actual correspondences
            pts1 = inl_mask[k][:, :2] # coordinates in image 1
            pts2 = inl_mask[k][:, 2:]  # coordinates in image 2
        else:
            pts1 = m[inl_mask[k].astype(bool),:2] # coordinates in image 1
            pts2 = m[inl_mask[k].astype(bool),2:]  # coordinates in image 2
        p1n = normalize_keypoints(pts1, K1).astype(np.float64)
        p2n = normalize_keypoints(pts2, K2).astype(np.float64)
        ang_errors[k] = max(eval_essential_matrix(p1n, p2n, E.reshape(3,3), dR, dT))
    return ang_errors



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
    
    if args.method.lower() not in ['cv2e', 'nmnet2', 'sklearn', 'cne', 'acne', 'load_oanet', 'load_oanet_ransac']:
        raise ValueError('Unknown value for --method')
    NUM_RUNS = 1
    if args.split == 'test':
        NUM_RUNS = 3
    if args.method.lower()  in ['nmnet2', 'cne', 'acne']:
        NUM_RUNS=1
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
    all_maas = []
    for run in range(NUM_RUNS):
        seqs = os.listdir(IN_DIR)
        for seq in seqs:
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
                error = evaluate_results(IN_DIR, seq,  models, inlier_masks)
            save_h5(error, out_errors_fname)
            mAA = calc_mAA_FE({seq: error})
            print (f" mAA {seq} = {mAA[seq]:.5f}")
            save_h5({"mAA": mAA[seq]}, out_maa_fname)
            all_maas.append(mAA[seq])
    out_maa_final_fname = os.path.join(OUT_DIR, f'maa_FINAL.h5')
    final_mAA = (np.array(all_maas)).mean()
    print (f" mAA total = {final_mAA:.5f}")
    save_h5({"mAA": final_mAA}, out_maa_final_fname)
    print ('Done!')
        


