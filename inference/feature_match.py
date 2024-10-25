
from __future__ import print_function

import os
import argparse
import configparser
from os.path import join
from os.path import exists
import torch
import numpy as np
import faiss
from tqdm.auto import tqdm
from scipy.spatial.distance import cdist
import sys
sys.path.append('../')
from datasets_original import PlaceDataset


def LayersDist(qv_pool1,dbv_pool1, qv_pool2,dbv_pool2):
    dist_pool1=cdist(qv_pool1, dbv_pool1, 'minkowski', p=2.)
    dist_pool2=cdist(qv_pool2, dbv_pool2, 'minkowski', p=2.)
    dist=dist_pool1+dist_pool2      
    #dist=torch.tensor(dist)  
    return dist

def compute_recall(gt, predictions, numQ, n_values, recall_str=''):
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / numQ
    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        tqdm.write("====> Recall {}@{}: {:.4f}".format(recall_str, n, recall_at_n[i]))
    return all_recalls


def write_kapture_output(opt, eval_set, predictions, outfile_name):
    if not exists(opt.result_save_folder):
        os.mkdir(opt.result_save_folder)
    outfile = join(opt.result_save_folder, outfile_name)
    print('Writing results to', outfile)
    with open(outfile, 'w') as kap_out:
        kap_out.write('# kapture format: 1.0\n')
        kap_out.write('# query_image, map_image\n')
        image_list_array = np.array(eval_set.images)
        for q_idx in range(len(predictions)):
            full_paths = image_list_array[predictions[q_idx]]
            query_full_path = image_list_array[eval_set.numDb + q_idx]
            for ref_image_name in full_paths:
                kap_out.write(query_full_path + ', ' + ref_image_name + '\n')


def write_recalls_output(opt, recalls_netvlad, n_values):
    if not exists(opt.result_save_folder):
        os.mkdir(opt.result_save_folder)
    outfile = join(opt.result_save_folder, 'recalls.txt')
    print('Writing recalls to', outfile)
    with open(outfile, 'w') as rec_out:
        for n in n_values:
            rec_out.write("Recall {}@{}: {:.4f}\n".format('MS-NetVLAD', n, recalls_netvlad[n]))
        #for n in n_values:
        #    rec_out.write("Recall {}@{}: {:.4f}\n".format('PatchNetVLAD', n, recalls_patchnetvlad[n]))


def feature_match(eval_set, device, opt, config):

    input_query_global_features_prefix1 = join(opt.query_input_features_dir, 'qry_globalfeats_pool1.npy')
    input_query_global_features_prefix2 = join(opt.query_input_features_dir, 'qry_globalfeats_pool2.npy')

    input_index_global_features_prefix1 = join(opt.index_input_features_dir, 'db_globalfeats_pool1.npy')
    input_index_global_features_prefix2 = join(opt.index_input_features_dir, 'db_globalfeats_pool2.npy')

    qFeat1 = np.load(input_query_global_features_prefix1)
    qFeat2 = np.load(input_query_global_features_prefix2)
    pool_size1 = qFeat1.shape[1]
    pool_size2 = qFeat2.shape[1]
    dbFeat1 = np.load(input_index_global_features_prefix1)
    dbFeat2 = np.load(input_index_global_features_prefix2)

    if dbFeat1.dtype != np.float32 or dbFeat2.dtype != np.float32:
        qFeat1 = qFeat1.astype('float32')
        dbFeat1 = dbFeat1.astype('float32')
        qFeat2 = qFeat2.astype('float32')
        dbFeat2 = dbFeat2.astype('float32')

    qFeat_cat = np.concatenate((qFeat1,qFeat2),axis=1)
    dbFeat_cat = np.concatenate((dbFeat1,dbFeat2),axis=1)
    tqdm.write('====> Building faiss index')
    pool_size_cat = qFeat_cat.shape[1]
   

    n_values = []
    for n_value in config['feature_match']['n_values_all'].split(","):  # remove all instances of n that are bigger than maxK
        n_values.append(int(n_value))

    tqdm.write('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size2)    
    faiss_index.add(dbFeat2)    
    

    if config['feature_match']['pred_input_path'] != 'None':
        predictions = np.load(config['feature_match']['pred_input_path'])  # optionally load predictions from a np file
    else:
        if opt.ground_truth_path and 'tokyo247' in opt.ground_truth_path:
            print('Tokyo24/7: Selecting only one of the 12 cutouts per panorama')
            # followed nnSearchPostprocess in https://github.com/Relja/netvlad/blob/master/datasets/dbTokyo247.m
            # noinspection PyArgumentList
            _, predictions = faiss_index.search(qFeat2, max(n_values) * 12)  # 12 cutouts per panorama
            predictions_new = []
            for qIx, pred in enumerate(predictions):
                _, idx = np.unique(np.floor(pred / 12).astype(np.int_), return_index=True)
                pred = pred[np.sort(idx)]
                pred = pred[:max(n_values)]
                predictions_new.append(pred)
            predictions = np.array(predictions_new)
        else:
            # noinspection PyArgumentList
            _, predictions = faiss_index.search(qFeat2, min(len(dbFeat2), max(n_values)))
   
    write_kapture_output(opt, eval_set, predictions, 'MSNV_cph_predictions.txt')


    print('Finished matching features.')

    # for each query get those within threshold distance
    if opt.ground_truth_path is not None:
        print('Calculating recalls using ground truth.')
        gt = eval_set.get_positives()

        global_recalls = compute_recall(gt, predictions, eval_set.numQ, n_values, 'MS-NetVLAD')

        write_recalls_output(opt, global_recalls, n_values)
    else:
        print('No ground truth was provided; not calculating recalls.')


def main():
    parser = argparse.ArgumentParser(description='Patch-NetVLAD-Feature-Match')
    parser.add_argument('--config_path', type=str, default='./PCA_Tools/configs/performance.ini',
                        help='File name (with extension) to an ini file that stores most of the configuration data for patch-netvlad')
    parser.add_argument('--dataset_root_dir', type=str, default='/home/Mapillary/',
                        help='If the files in query_file_path and index_file_path are relative, use dataset_root_dir as prefix.')
    parser.add_argument('--query_file_path', type=str, default='path to query .txt file',
                        help='Path (with extension) to a text file that stores the save location and name of all query images in the dataset')
    parser.add_argument('--index_file_path', type=str, default='/mapillarycph_imageNames_index.txt',
                        help='path to reference database .txt file')
    parser.add_argument('--query_input_features_dir', type=str, default='/save/',
                        help='Path to load all query features')
    parser.add_argument('--index_input_features_dir', type=str, default='/save/',
                        help='Path to load all database features')
    parser.add_argument('--ground_truth_path', type=str, default=None,
    #parser.add_argument('--ground_truth_path', type=str, default=None,
                        help='Path (with extension) to a file that stores the ground-truth data; ')
    parser.add_argument('--result_save_folder', type=str, default='/results/')
    parser.add_argument('--posDistThr', type=int, default=None, help='Manually set ground truth threshold')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')

    opt = parser.parse_args()
    print(opt)
    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)

    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")

    #if not os.path.isfile(opt.query_file_path):
    #    opt.query_file_path = join(PATCHNETVLAD_ROOT_DIR, 'dataset_imagenames', opt.query_file_path)
    #if not os.path.isfile(opt.index_file_path):
    #    opt.index_file_path = join(PATCHNETVLAD_ROOT_DIR, 'dataset_imagenames', opt.index_file_path)
        
    dataset = PlaceDataset(opt.query_file_path, opt.index_file_path, opt.dataset_root_dir, opt.ground_truth_path,
                           config['feature_extract'])#, posDistThr=opt.posDistThr)
    
    feature_match(dataset, device, opt, config)

    #torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
                              # memory after runs
    print('Done')


if __name__ == "__main__":
    main()
