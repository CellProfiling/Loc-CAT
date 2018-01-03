#!/usr/bin/python3
import localizer
from dataset import read_if_file, get_if_info_classes
from dataset import read_feat_file
from dataset import get_plate_ids
from train_dnn import DEFAULT_INDICES
import argparse
import glob
import os.path
import pickle

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model_file')
    argparser.add_argument('if_file')
    argparser.add_argument('test_folder')
    argparser.add_argument('save_output_path')
    args = argparser.parse_args()

    test_folder = args.test_folder
    test_files = glob.glob(os.path.join(test_folder, '*-testing-*'))
    test_files = [f for f in test_files if not f.endswith('-0')]

    model_file = args.model_file
    localizer = localizer.ProteinLocalizer(path=model_file)

    if_file = args.if_file
    if_info = read_if_file(if_file, split_to_well=False)
    if_info_classes = get_if_info_classes(if_info)

    output_path = args.save_output_path

    for i, test_file in enumerate(sorted(test_files)):
        print(test_file)
        f_gen = read_feat_file(test_file, DEFAULT_INDICES, if_info)
        p_gen = get_plate_ids(test_file, DEFAULT_INDICES, if_info)

        predictions = localizer.predict_fovs((f_gen, p_gen), False, False, external=True, apply_cutoffs=False)
        output_file = open(output_path + '-' + str(i+1), 'wb')
        pickle.dump(predictions, output_file)
        output_file.close()
