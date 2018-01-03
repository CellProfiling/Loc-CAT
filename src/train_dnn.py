#!/usr/bin/python3
import localizer
from dataset import read_if_file, get_if_info_classes
from dataset import get_binary_classes, read_feat_file
from dataset import get_plate_ids
import argparse
import pickle
import configparser
import logging
import os
import glob
import csv
import tempfile

DEFAULT_INDICES = list(range(7, 720)) + [2210, 2214, 2218, 2222, 2226, 2230]


def create_training_validation_files(training_file, well_file):
    training_file = csv.reader(open(training_file))
    wells = pickle.load(open(well_file, 'rb'))
    training_wells = list(wells['training'])
    num_val_wells = int(0.10 * len(training_wells))
    validation_wells = set(training_wells[:num_val_wells])
    training_wells = set(training_wells[num_val_wells:])

    train_temp = tempfile.NamedTemporaryFile(mode='w', delete=False)
    val_temp = tempfile.NamedTemporaryFile(mode='w', delete=False)

    train_writer = csv.writer(train_temp)
    val_writer = csv.writer(val_temp)
    for line in training_file:
        well_id = '_'.join(line[0].split('_')[:-1])
        if well_id in validation_wells:
            val_writer.writerow(line)
        if well_id in training_wells:
            train_writer.writerow(line)
    return train_temp.name, val_temp.name


def cross_validate(data_folder, if_info, if_info_classes, cfgparser,
                   output_path=None, model_name=None,
                   locations_column='locations'):
    for i in range(5):
        training_file = os.path.join(data_folder, '*-training-' + str(i))
        training_file = glob.glob(training_file)[0]
        testing_file = os.path.join(data_folder, '*-testing-' + str(i))
        testing_file = glob.glob(testing_file)[0]
        well_file = os.path.join(data_folder, '*-wells-' + str(i))
        well_file = glob.glob(well_file)[0]

        tv = create_training_validation_files(training_file, well_file)
        training_file, validation_file = tv

        l_gen = get_binary_classes(training_file, if_info, if_info_classes,
                                   DEFAULT_INDICES, locations_column)
        f_gen = read_feat_file(training_file, DEFAULT_INDICES, if_info, locations_column)

        vl_gen = get_binary_classes(validation_file, if_info, if_info_classes,
                                    DEFAULT_INDICES, locations_column)
        vf_gen = read_feat_file(validation_file, DEFAULT_INDICES, if_info, locations_column)

        predictor = localizer.ProteinLocalizer(cfgparser)
        predictor.train_model((f_gen, l_gen), epochs=500,
                              validation=(vf_gen, vl_gen), verbose=True,
                              early_stopping=10)

        test_l_gen = get_binary_classes(testing_file, if_info, if_info_classes,
                                        DEFAULT_INDICES, locations_column)
        test_f_gen = read_feat_file(testing_file, DEFAULT_INDICES, if_info, locations_column)
        test_p_gen = get_plate_ids(testing_file, DEFAULT_INDICES, if_info, locations_column)

        if output_path:
            prediction = predictor.predict_fovs((test_f_gen, test_p_gen),
                                                apply_cutoffs=False,
                                                external=True)
            save_file = open(output_path + '-' + str(i), 'wb')
            pickle.dump(prediction, save_file)

        if model_name:
            predictor.save(model_name + '-' + str(i))

        res = predictor.test_model((test_f_gen, test_l_gen, test_p_gen))
        print('Cell, 0.4 cutoff:', res[0])
        print('FOV hamming, 0.4 cutoff:', res[1])
        del(predictor)
        del(training_file)
        del(validation_file)


if __name__ == '__main__':
    logging.debug('Parsing arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument('data_folder')
    parser.add_argument('if_file')
    parser.add_argument('localizer_config_file')
    parser.add_argument('--location-column', default='locations')
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--save-model', required=True)
    args = parser.parse_args()

    data_folder = args.data_folder
    if_file = args.if_file
    config_file = args.localizer_config_file
    output_path = args.output_path
    model_name = args.save_model

    logging.info('Reading config file')
    cfgparser = configparser.ConfigParser()
    cfgparser.readfp(open(config_file))

    logging.info('Reading IF file')
    if_info = read_if_file(if_file, split_to_well=False)

    location_column = args.location_column
    if_info_classes = get_if_info_classes(if_info, location_column)
    print(len(if_info_classes)/2, 'classes')

    cross_validate(data_folder, if_info, if_info_classes, cfgparser,
                   output_path, model_name, location_column)
