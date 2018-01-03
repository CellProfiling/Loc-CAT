#!/usr/bin/python3
from iter_utils import repeatable_generator
import os
import csv
import collections
import logging
import sys
import argparse
import pickle
import numpy as np
from sklearn.model_selection import KFold as KF

logging.basicConfig(filename='if_dataset.log', level=logging.INFO)


@repeatable_generator
def read_feat_file(filename, indices, if_info, locations_header='locations'):
    f = csv.reader(open(filename))
    feat_length = None
    for line in f:
        if line[0] not in if_info:
            continue
        if 'NULL' in if_info[line[0]][locations_header]:
            continue
        if 'Negative' in if_info[line[0]][locations_header]:
            continue
        if 'Unspecific' in if_info[line[0]][locations_header]:
            continue

        line = line[1:]
        if not feat_length:
            feat_length = len(line)
        if len(line) != feat_length:
            continue

        feats = [float(line[index]) for index in indices]
        if any(np.isnan(feats)):
            continue
        yield feats


@repeatable_generator
def get_plate_ids(filename, indices, if_info, locations_header='locations'):
    f = csv.reader(open(filename))
    feat_length = None
    for line in f:
        if not feat_length:
            feat_length = len(line)
        if len(line) != feat_length:
            continue
        feats = [float(line[index]) for index in indices]
        if any(np.isnan(feats)):
            continue
        id_ = line[0]
        if id_ not in if_info:
            continue
        if 'NULL' in if_info[id_][locations_header]:
            continue
        if 'Negative' in if_info[id_][locations_header]:
            continue
        if 'Unspecific' in if_info[id_][locations_header]:
            continue
        yield id_


@repeatable_generator
def get_binary_classes(filename, if_info, if_info_classes,
                       indices, locations_header='locations'):
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
        if id_ not in if_info:
            continue
        if 'NULL' in if_info[id_][locations_header]:
            continue
        if 'Negative' in if_info[id_][locations_header]:
            continue
        if 'Unspecific' in if_info[id_][locations_header]:
            continue

        if_classes = if_info[id_][locations_header]
        bin_classes = convert_to_binary_classes(if_classes, if_info_classes)
        yield bin_classes


def convert_to_binary_classes(string_classes, if_info_classes):
    string_classes = string_classes.split(',')
    bin_classes = [0] * int(len(if_info_classes)/2)
    for c in string_classes:
        bin_classes[if_info_classes[c]] = 1
    return bin_classes


def get_if_info_classes(if_info, locations_header='locations'):
    classes = set()
    for p in if_info:
        unsplit = if_info[p][locations_header]
        split = unsplit.split(',')
        classes.update(split)
    classes = sorted(classes)
    ordered_classes = collections.OrderedDict()
    for i, c in enumerate(classes):
        # in the unlikely event that a class is an int
        # (not a str representation of int)
        if isinstance(c, int):
            logging.critical('Class CANNOT be an int')
            raise ValueError('{} is int when it should be str'.format(c))

        ordered_classes[c] = i
        ordered_classes[i] = c
    return ordered_classes



def read_if_file(if_file, include={}, non_include={}, identifiers=['if_plate_id', 'position', 'sample'],
                 split_to_well=True):
    dictreader = csv.DictReader(open(if_file))
    if_info = dict()
    for line in dictreader:
        skipping = False

        for constraint in include:
            actual_line = line[constraint]
            actual_line = actual_line.split(',')

            any_in = [actual in include[constraint] for actual in actual_line]
            if not any(any_in):
                skipping = True
                break

        if skipping:
            continue

        for constraint in non_include:
            actual_line = line[constraint]
            value = actual_line.split(',')

            any_in = [actual in non_include[constraint] for actual in value]
            if any(any_in):
                skipping = True
                break

        if skipping:
            continue

        id_ = ''
        for i in identifiers:
            id_ += line[i] + '_'

        if split_to_well:
            id_ = '_'.join(id_.split('_')[:-2])
        else:
            id_ = '_'.join(id_.split('_')[:-1])
        if_info[id_] = line
    return if_info


def split_arguments(arg_list):
    args = collections.defaultdict(set)
    for arg in arg_list:
        arg = arg.split(':')
        if len(arg) > 2:
            logging.critical('An input constraint has to be on the form:')
            logging.critical('<IF_COLUMN>:<CONSTRAINT>')
            logging.critical('Exiting!')
            sys.exit(-1)
        argname = arg[0].strip()
        argconstraint = arg[1].strip()
        args[argname].add(argconstraint)
    return args


def read_existance_check(existance_file):
    existance_file = open(existance_file, 'r')
    stuff = set()
    for line in csv.reader(existance_file):
        stuff.add('_'.join(line[0].split('_')[:-2]))
    return stuff


if __name__ == '__main__':
    masterfeatmat = '/home/casper/Documents/HDD/platefeatmats2/{}masterfeatmat.csv'
    argparser = argparse.ArgumentParser()
    argparser.add_argument("if_file")
    argparser.add_argument("output_file")
    argparser.add_argument("--include", default=list(), action="append")
    argparser.add_argument("--non-include", default=list(), action="append")
    # For stuff like gamer.
    # Check the 0th item on each line
    argparser.add_argument("--check-existance-in")
    args = argparser.parse_args()

    if args.check_existance_in:
        existance_check = read_existance_check(args.check_existance_in)
    else:
        existance_check = None

    include_args = split_arguments(args.include)
    non_include_args = split_arguments(args.non_include)

    if_info = read_if_file(args.if_file, include_args, non_include_args)
    pickle.dump(if_info, open(args.output_file + '-if_info', 'wb'))
    wells = set(if_info.keys())
    if existance_check:
        print(len(wells))
        wells = wells.intersection(existance_check)
        print(len(wells))
    wells = np.asarray(list(wells), dtype=str)
    np.random.shuffle(wells)
    del(existance_check)

    kfold = KF(n_splits=5, shuffle=False).split(wells)
    for i, (training, testing) in enumerate(kfold):
        logging.info('Current iteration: {}'.format(i))
        output_file = open(args.output_file + '-wells-' + str(i), 'wb')
        pickle.dump({'training': wells[training], 'testing': wells[testing]}, output_file)
        output_file.close()

        training_plates = {well.split('_')[0] for well in wells[training]}
        training_wells = {well for well in wells[training]}

        testing_plates = {well.split('_')[0] for well in wells[testing]}
        testing_wells = {well for well in wells[testing]}

        training_cells = []
        testing_cells = []

        all_plates = training_plates.union(testing_plates)
        training_file_name = args.output_file + '-training-' + str(i)
        testing_file_name = args.output_file + '-testing-' + str(i)
        training_file = open(training_file_name, 'w')
        testing_file = open(testing_file_name, 'w')
        for plate in sorted(all_plates):
            logging.info('Current plate: {}'.format(plate))
            reader = csv.reader(open(masterfeatmat.format(plate)))
            for line in reader:
                filename = line[0]
                filename = '_'.join(filename.split('_')[:-1])
                logging.debug(filename)
                if filename in training_wells:
                    training_cells.append(line)
                if filename in testing_wells:
                    testing_cells.append(line)

            if len(training_cells) > 25000:
                for cell in training_cells:
                    for cell_attribute in cell:
                        training_file.write(cell_attribute)
                        training_file.write(',')
                    training_file.write('\b\n')
                training_cells = []

                for cell in testing_cells:
                    for cell_attribute in cell:
                        testing_file.write(cell_attribute)
                        testing_file.write(',')
                    testing_file.write('\b\n')
                testing_cells = []

        for cell in training_cells:
            for cell_attribute in cell:
                training_file.write(cell_attribute)
                training_file.write(',')
            training_file.write('\b\n')
        training_file.close()

        for cell in testing_cells:
            for cell_attribute in cell:
                testing_file.write(cell_attribute)
                testing_file.write(',')
            testing_file.write('\b\n')
        testing_file.close()

        os.system('sort --random-sort "{0}" --random-source /dev/urandom > "{0}_random"'.format(training_file_name))
        os.system('mv "{0}_random" "{0}"'.format(training_file_name))
        os.system('sort --random-sort "{0}" --random-source /dev/urandom > "{0}_random"'.format(testing_file_name))
        os.system('mv "{0}_random" "{0}"'.format(testing_file_name))
