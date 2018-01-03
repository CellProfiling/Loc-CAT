import argparse
from proteinlocalizer import dnn
from proteinlocalizer.utils.data_utils import external_zscore
import tensorflow as tf
import csv
import dataset
import glob
import pickle
import os.path
import configparser
import numpy as np
from train_dnn import create_training_validation_files
from proteinlocalizer.utils.iter_utils import repeatable_generator
from proteinlocalizer import localizer

DEFAULT_INDICES = list(range(7, 720)) + [2210, 2214, 2218, 2222, 2226, 2230]


@repeatable_generator
def read_feat_file_with_pvals(filename, pvals, indices, include_the_pvals=True):
    f = csv.reader(open(filename))
    feat_length = None
    for line in f:
        line_id = line[0]
        line = line[1:]
        if not feat_length:
            feat_length = len(line)
        if len(line) != feat_length:
            continue

        feats = [float(line[index]) for index in indices]
        if any(np.isnan(feats)):
            continue
        if line_id not in pvals:
            continue

        if include_the_pvals:
            feats.extend(pvals[line_id])
        yield feats


@repeatable_generator
def get_pvals_using_file_ids(filename, pvals, indices):
    f = csv.reader(open(filename))
    feat_length = None
    for line in f:
        line_id = line[0]
        line = line[1:]
        if not feat_length:
            feat_length = len(line)
        if len(line) != feat_length:
            continue

        feats = [float(line[index]) for index in indices]
        if any(np.isnan(feats)):
            continue
        if line_id not in pvals:
            continue

        yield pvals[line_id]


@repeatable_generator
def get_binary_classes_that_have_pvals(filename, if_info, if_info_classes,
                                       indices, pvals, locations_header='locations'):
    """
    if_info should have the same ids as the first column in the feat_file

    if_info_classes should have the exact same format as the object obtained
    from `get_if_info_classes`, i.e. a dict with both str->index and index->str
    """
    f = csv.reader(open(filename))
    feat_length = None
    for line in f:
        # I have to do this somewhere and I don't want to deal with the crap
        # that happened last time I tried combining feats and label output.
        if not feat_length:
            feat_length = len(line)
        if len(line) != feat_length:
            continue
        feats = [float(line[index]) for index in indices]
        if any(np.isnan(feats)):
            continue

        id_ = line[0]
        if id_ not in pvals:
            continue
        if_classes = if_info[id_][locations_header]
        bin_classes = convert_to_binary_classes(if_classes, if_info_classes)
        yield bin_classes


@repeatable_generator
def get_plate_ids_with_pvals(filename, indices, pvals):
    f = csv.reader(open(filename))
    feat_length = None
    for line in f:
        # I have to do this somewhere and I don't want to deal with the crap
        # that happened last time I tried combining feats and label output.
        if not feat_length:
            feat_length = len(line)
        if len(line) != feat_length:
            continue
        feats = [float(line[index]) for index in indices]
        if any(np.isnan(feats)):
            continue
        id_ = line[0]
        if id_ not in pvals:
            continue
        yield id_


def convert_to_binary_classes(string_classes, if_info_classes):
    string_classes = string_classes.split(',')
    bin_classes = [0] * int(len(if_info_classes)/2)
    for c in string_classes:
        bin_classes[if_info_classes[c]] = 1
    return bin_classes


def read_pval_file(pval_file):
    pvals = {}
    reader = csv.reader(open(pval_file, 'r'))
    next(reader)
    for line in reader:
        line_pvals = [float(f) for f in line[1:]]
        line_id = line[0].rstrip('_')
        pvals[line_id] = line_pvals
    return pvals


def train_pval_network(training_file, pvals, indices, save_path):
    pval_network = dnn.DNN()
    pval_network.add_layer([719, 200], dropout=0.2)
    pval_network.add_layer([200, 100], tf.nn.relu6, dropout=0.4)
    pval_network.add_layer([100, 29], tf.nn.sigmoid)
    pval_network.build(cost=lambda x, y: tf.square(x-y))

    f_gen = read_feat_file_with_pvals(training_file, pvals, indices, include_the_pvals=False)
    pval_gen = get_pvals_using_file_ids(training_file, pvals, indices)

    f_gen, mean, stddev = external_zscore(f_gen)
    pval_network.train((f_gen, pval_gen), epochs=100, batch_size=1000, verbose=True)
    pval_network.save(save_path)
    print('Pval cost:', pval_network.cost((f_gen, pval_gen)))
    return pval_network, mean, stddev


@repeatable_generator
def extend_feats_with_predicted_pvals(f_gen, pval_network, mean, stddev):
    f_gen_2, _, _ = external_zscore(f_gen, mean, stddev)
    for f, p in zip(f_gen, pval_network.predict(f_gen_2)):
        f.extend(p)
        yield f


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder')
    parser.add_argument('if_file')
    parser.add_argument('localizer_config_file')
    parser.add_argument('pval_file')
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--model-path', required=True)
    parser.add_argument('--predict-gamers', action='store_true')
    args = parser.parse_args()

    if_file = args.if_file
    if_info = dataset.read_if_file(if_file, split_to_well=False)
    if_info_classes = dataset.get_if_info_classes(if_info)

    data_folder = args.data_folder
    training_files = sorted(glob.glob(os.path.join(data_folder, '*-training-*')))
    testing_files = sorted(glob.glob(os.path.join(data_folder, '*-testing-*')))
    well_files = sorted(glob.glob(os.path.join(data_folder, '*-wells-*')))

    pval_file = args.pval_file
    pvals = read_pval_file(pval_file)

    config_file = args.localizer_config_file
    cfgparser = configparser.ConfigParser()
    cfgparser.readfp(open(config_file))

    output_path = args.output_path
    model_name = args.model_path

    predict_gamers = args.predict_gamers
    print(predict_gamers)

    for i, (training_file, testing_file, well_file) in enumerate(zip(training_files, testing_files, well_files)):
        print(training_file, testing_file)
        training_file, validation_file = create_training_validation_files(training_file, well_file)

        if predict_gamers:
            pval_network, mean, stddev = train_pval_network(training_file, pvals, DEFAULT_INDICES,
                                                            model_name + '-gamer-prediction-network-' + str(i))
            f_gen = read_feat_file_with_pvals(training_file, pvals, DEFAULT_INDICES, include_the_pvals=False)
            f_gen = extend_feats_with_predicted_pvals(f_gen, pval_network, mean, stddev)

            vf_gen = read_feat_file_with_pvals(validation_file, pvals, DEFAULT_INDICES, include_the_pvals=False)
            vf_gen = extend_feats_with_predicted_pvals(vf_gen, pval_network, mean, stddev)

            tf_gen = read_feat_file_with_pvals(testing_file, pvals, DEFAULT_INDICES, include_the_pvals=False)
            tf_gen = extend_feats_with_predicted_pvals(tf_gen, pval_network, mean, stddev)
        else:
            f_gen = read_feat_file_with_pvals(training_file, pvals, DEFAULT_INDICES)
            vf_gen = read_feat_file_with_pvals(validation_file, pvals, DEFAULT_INDICES)
            tf_gen = read_feat_file_with_pvals(testing_file, pvals, DEFAULT_INDICES)

        l_gen = get_binary_classes_that_have_pvals(training_file, if_info, if_info_classes, DEFAULT_INDICES, pvals)

        vl_gen = get_binary_classes_that_have_pvals(validation_file, if_info, if_info_classes, DEFAULT_INDICES, pvals)

        tl_gen = get_binary_classes_that_have_pvals(testing_file, if_info, if_info_classes, DEFAULT_INDICES, pvals)
        tp_gen = get_plate_ids_with_pvals(testing_file, DEFAULT_INDICES, pvals)

        predictor = localizer.ProteinLocalizer(cfgparser)
        predictor.train_model((f_gen, l_gen), validation=(vf_gen, vl_gen), epochs=300, verbose=True, early_stopping=10)

        res = predictor.test_model((tf_gen, tl_gen, tp_gen))
        print('Cell, 0.4 cutoff:', res[0])
        print('FOV hamming, 0.4 cutoff:', res[1])

        prediction = predictor.predict_fovs((tf_gen, tp_gen), apply_cutoffs=False, external=True)
        save_file = open(output_path + '-' + str(i), 'wb')
        pickle.dump(prediction, save_file)
        predictor.save(model_name + '-' + str(i))
        del(predictor)
        del(training_file)
        del(validation_file)
