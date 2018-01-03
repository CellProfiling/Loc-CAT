#!/usr/bin/python3
"""
Inputs (in order):
    model file: which model to use for prediction
    feature file: a file containing image features for predictions
    output file: where to output predictions for the images
    cutoff file: the cutoffs to use for prediction

Example usage:
    python3 prediction_demo.py ../models/dnn-model.model ../data/example_feature_file.csv output.txt ../data/cutoffs
"""
import argparse
import localizer
import csv
import numpy as np
from train_dnn import DEFAULT_INDICES


def get_cutoffs(path):
    cutoff_array = []
    cutoffs = []
    for line in open(path, 'r'):
        split = line.split(':')
        cutoff = float(split[1])
        if (cutoff < 0.0001):
            cutoff = 0.4
        cutoffs.append((split[0], cutoff))
        cutoff_array.append(cutoff)
    return cutoffs, cutoff_array


def feature_reader(filename, indices=DEFAULT_INDICES):
    f = csv.reader(open(filename))
    for line in f:
        id_ = line[0]
        line = line[1:]

        try:
            feats = [float(line[index]) for index in indices]
        except IndexError:
            print(id_, ' failed to index correctly. Skipping!')
        if any(np.isnan(feats)):
            print(id_, ' has NaNs. Skipping!')
            continue
        yield feats


def plate_id_reader(filename, indices=DEFAULT_INDICES):
    f = csv.reader(open(filename))
    for line in f:
        id_ = line[0]
        line = line[1:]

        try:
            feats = [float(line[index]) for index in indices]
        except IndexError:
            print(id_, ' failed to index correctly. Skipping!')
        if any(np.isnan(feats)):
            print(id_, ' has NaNs. Skipping!')
            continue

        yield id_


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('model_file')
    argparser.add_argument('feature_file')
    argparser.add_argument('output_file')
    argparser.add_argument('cutoff_file')
    args = argparser.parse_args()

    localizer = localizer.ProteinLocalizer(path=args.model_file)
    f_gen = feature_reader(args.feature_file)
    p_gen = plate_id_reader(args.feature_file)
    cutoffs, cutoff_array = get_cutoffs(args.cutoff_file)

    predictions = localizer.predict_fovs((f_gen, p_gen), False, False, external=True, apply_cutoffs=False)
    output_file = open(args.output_file, 'w')
    for p_id, prediction in predictions.items():
        prediction['prediction'][np.argmax(prediction['prediction'])] = 1
        pred = np.greater_equal(prediction['prediction'], cutoff_array)
        output_file.write(p_id)

        for i, p in enumerate(pred):
            if p:
                output_file.write(',')
                output_file.write(cutoffs[i][0])
        output_file.write('\n')
    output_file.close()
