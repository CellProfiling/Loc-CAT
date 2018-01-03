import dnn
import data_utils
import iter_utils
import collections
import tempfile
import zipfile
import os
import json
import tensorflow as tf
import numpy as np


ACTIVATION_FUNCTIONS = {
        'relu': tf.nn.relu,
        'relu6': tf.nn.relu6,
        'identity': tf.identity,
        'sigmoid': tf.nn.sigmoid,
        'softmax': tf.nn.softmax,
        }


def _activation_from_string(string):
    return ACTIVATION_FUNCTIONS[string]


class ProteinLocalizer(object):
    def __init__(self, settings=None, path=None, classes=None, seed=None):
        """
        Creates a ProteinLocalizer using the supplied settings.
        Args:
            settings: A configparser that has the section Localizer with the
            options n_in, n_out, n_neurons, n_hidden, zscore, model_path,
            retrain, activation, and out_activation.

            path: A path to a previously saved localizer.
                  Will automatically load the localizer using the load method.

            classes: The classes used when localizing.
                     If None, only binary output for predictions is available.

            seed: The seed for the PRNGs relevant for the localizer.
        """

        if (not settings and not path) or (settings and path):
            raise ValueError('Either settings or path must be supplied')

        if path and not os.path.isfile(path):
            raise ValueError('No model available at %s' % path)

        if not path:
            n_in = settings.getint('Localizer', 'n_in')
            n_out = settings.getint('Localizer', 'n_out')
            n_neurons = settings.getint('Localizer', 'n_neurons')
            n_hidden = settings.getint('Localizer', 'n_hidden')

            self._mean = None
            self._sigma = None
            self._zscore = settings.getboolean('Localizer', 'zscore')

            self._classes = classes

            activation_string = settings.get('Localizer', 'activation')
            out_activation = settings.get('Localizer', 'out_activation')
            activation = _activation_from_string(activation_string)
            out = _activation_from_string(out_activation)
            self.__build_dnn(n_in, n_out, n_neurons, n_hidden,
                             activation, out, seed)

        else:
            self.load(path)

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, classes):
        self._classes = classes

    def __build_dnn(self, nin, nout, nneurons, nhidden, activation=tf.nn.relu6,
                    outactivation=tf.nn.sigmoid, seed=None):
        """
        Builds a neural network based on the parameters and stores it in
        self.dnn.
        """
        self.dnn = dnn.DNN(seed=seed)
        self.dnn.add_layer([nin, nneurons], dropout=0.2)
        for _ in range(2, nhidden):
            self.dnn.add_layer([nneurons, nneurons], activation, dropout=0.4)
        self.dnn.add_layer([nneurons, nout], outactivation)
        self.dnn.build()

    def train_model(self, data, epochs=1000, batch_size=1000,
                    validate=None, verbose=False, early_stopping=None,
                    validation=None):
        """
        Trains a model on the supplied data for the desired number of epochs
        and batch sizes.

        Args:
            data: A tuple containing features and labels.
                  The labels must be supplied as binary vectors, and the
                  features as numerical values.

            epochs: How many epochs the model should be trained for.

            batch_size: How many items should be used at a time.
                        The more memory there is available, the larger this
                        value can be.
            validate: A float  between 0 and 1 determining how much data should
                      be used for a validation set. If None, no validation set
                      is used.
                      The fraction for validation is taken from the first items
                      of the dataset.

            verbose: Set verbose training on or off.

            early_stopping: Determines if early stopping should be used.
                            If None, no early stopping will be used.
                            If it is an integer, training will stop after
                            `early_stopping` epochs of non-decreasing
                            validation cost.

                            Does not have an effect if there is no validation
                            set.
        """
        if self._zscore:
            if verbose:
                print('Normalizing data, may take some time')
            zscored = data_utils.external_zscore(data[0])
            (feat_gen, self._mean, self._sigma) = zscored

        labelgen = data[1]

        val_data = None
        if validate:
            if verbose:
                print('Creating a validation set, may take some time')
            if not isinstance(validate, float) or not 0.0 < validate < 1.0:
                raise ValueError('validate needs to be a float')
            feat_gen, feat_len = iter_utils.iter_len(feat_gen)
            val_len = int(validate * feat_len)
            feat_gen, vfgen = data_utils.external_split(feat_gen, val_len)
            labelgen, vlgen = data_utils.external_split(labelgen, val_len)
            vfgen = iter_utils.repeat_gen(vfgen)
            vlgen = iter_utils.repeat_gen(vlgen)
            val_data = (vfgen, vlgen)

        if validation:
            vfgen = validation[0]
            if self._zscore:
                vfgen, _, _ = data_utils.external_zscore(vfgen, self._mean, self._sigma)
            val_data = (vfgen, validation[1])

        feat_gen = iter_utils.repeat_gen(feat_gen)
        labelgen = iter_utils.repeat_gen(labelgen)

        data = (feat_gen, labelgen)

        return self.dnn.train(data, validation_data=val_data, epochs=epochs,
                              batch_size=batch_size, verbose=verbose,
                              early_stopping=early_stopping)

    def test_model(self, data, cell_cutoff=0.4, verbose=False, fov_cutoff=0.4,
                   force_fov_prediction=False, force_cell_prediction=False):
        """
        Test the model on the supplied data.
        Tests the model on a per image basis, as well as per cell, using
        the majority cell classifications as the image classification.

        Args:
            data: A tuple containing feats, labels, and image names.
                  The tuple should be organized as (feats, labels,
                  image).
            verbose: Verbose testing on or off.

        Returns:
            The average cell hamming score, the average fov hamming score, the
            cost over the test set, the precision over the test set, and the
            recall.
        """
        x_data = data[0]
        y_data = data[1]
        plate_data = data[2]

        cell_predictions = self.predict_cells(x_data, force_cell_prediction,
                                              external=True)
        cell_predictions = np.asarray([x['prediction'] for x in
                                      cell_predictions])
        cell_predictions[cell_predictions > cell_cutoff] = 1
        cell_predictions[cell_predictions < 1] = 0
        cacc = data_utils.hamming_score(y_data, cell_predictions)

        fov_predictions = self.predict_fovs((x_data, plate_data),
                                            force_fov_prediction,
                                            force_cell_prediction,
                                            cell_cutoff, external=True)
        plabels = {}
        for p, l in zip(plate_data, y_data):
            if p in plabels:
                continue
            plabels[p] = l
        predictions = []
        labels = []
        for p in fov_predictions:
            predictions.append(fov_predictions[p]['prediction'])
            labels.append(plabels[p])
        predictions = np.asarray(predictions)
        predictions[predictions > fov_cutoff] = 1
        predictions[predictions < 1] = 0

        hamming_score = data_utils.hamming_score(labels, predictions)
        if self._zscore:
            z = data_utils.external_zscore(x_data, self._mean, self._sigma)
            (x_data, _, _) = z
        cost = self.dnn.cost((x_data, y_data))
        precision, recall = data_utils.precision_recall(labels, predictions)

        return cacc, hamming_score, cost, precision, recall

    def predict_cells(self, data, force_prediction=False, external=False,
                      localization_info=None):
        """
        Predicts on the given data and returns a list of predictions.
        The predictions are returned as floating point numbers.

        Params:
            data: An iterator of cell features to be classified.
            force_prediction: A boolean that determines if a guess should be
                              forced or not when no label could be predicted.
                              The label with the highest probability is set to
                              1 if this is True.
        Returns:
            A list of floating point numbers that represent the predictions
            for the input data.
        """
        if localization_info:
            localization_info = iter(localization_info)
        if self._zscore:
            if external:
                z = data_utils.external_zscore(data, self._mean, self._sigma)
                (data, _, _) = z
            else:
                z = data_utils.zscore(data, self._mean, self._sigma)
                (data, _, _) = z
        predictions = self.dnn.predict(data)
        if force_prediction:
            argmax = np.argmax(predictions, axis=1)
            for i in range(len(argmax)):
                predictions[i][argmax[i]] = 1

        for i in range(len(predictions)):
            predictions[i] = {'prediction': list(predictions[i])}
            if localization_info:
                predictions[i]['location'] = next(localization_info)
        return predictions

    def predict_fovs(self, data, force_fov_predictions=False,
                     force_cell_predictions=False, cell_cutoff=0.4,
                     external=False, apply_cutoffs=True):
        """
        Predicts the labels on a field of view level based on the
        classification on a cell level.

        Args:
            data: A tuple containing iterables of feats and plate names,
                  (feats, plate_names)
            force_fov_predictions: A boolean determining if a prediction should
                                   be forced on the FOV level.
                                   The label with the highest probability is
                                   set to one if this is True.
            force_cell_prediction: A boolean determining if a prediction should
                                   be forced on the cell level.
                                   The label with the highest probability is
                                   set to one for each cell if this is True.
            cell_cutoff:    The cutoff value for when a label is considered
                            True on the cell level.
        Returns:
            An OrderedDict containing the plate names mapped to the FOV
            predictions. The predictions consists of floating point numbers
            determining the proportion of cells being predicted for each label.
        """
        feats = data[0]
        pgen = data[1]
        if len(data) > 2:
            positional = data[2]
        else:
            positional = None
        cell_predictions = self.predict_cells(feats, force_cell_predictions,
                                              external=external,
                                              localization_info=positional)

        # cell_predictions = np.asarray(cell_predictions)
        # cell_predictions[cell_predictions > cell_cutoff] = 1
        # cell_predictions[cell_predictions < 1] = 0

        plates = {}
        for (pred, plate) in zip(cell_predictions, pgen):
            if 'location' in pred:
                loc = pred['location']
            else:
                loc = ''
            pred = np.asarray(pred['prediction'])
            if apply_cutoffs:
                pred[pred > cell_cutoff] = 1
                pred[pred < 1] = 0

            if plate not in plates:
                plates[plate] = [pred, 1, []]
            else:
                list_ = plates[plate]
                list_[0] = np.add(list_[0], pred)
                list_[1] += 1

                cell = {'prediction': pred.tolist()}
                if loc:
                    cell['location'] = loc
                list_[2].append(cell)

        predictions = collections.OrderedDict()
        for plate in sorted(plates.keys()):
            predictions[plate] = {
                'prediction': np.divide(plates[plate][0], plates[plate][1]),
                'cells': plates[plate][2]
                }

        for pred in predictions:
            p = predictions[pred]['prediction']
            if force_fov_predictions:
                p[np.argmax(p)] = 1
            predictions[pred]['prediction'] = p.tolist()
        return predictions

    def predict_wells(self, data, force_well_prediction=False,
                      force_fov_prediction=False, force_cell_prediction=False,
                      cell_cutoff=0.4, fov_cutoff=0.4, external=False,
                      apply_cutoffs=True):
        fov_predictions = self.predict_fovs(data, force_fov_prediction,
                                            force_cell_prediction,
                                            cell_cutoff, external=external,
                                            apply_cutoffs=apply_cutoffs)
        w_predictions = {}
        for fov in fov_predictions:
            well = fov.split('_')
            well = well[0] + '_' + well[1]

            fov_pred = fov_predictions[fov]['prediction']
            fov_pred = np.asarray(fov_pred)
            if apply_cutoffs:
                fov_pred[fov_pred > fov_cutoff] = 1
                fov_pred[fov_pred < 1] = 0

            if well not in w_predictions:
                w_predictions[well] = [fov_predictions[fov]['prediction'], 1]
                w_predictions[well].append({})
            else:
                list_ = w_predictions[well]
                list_[0] = np.add(list_[0], fov_predictions[fov]['prediction'])
                list_[1] += 1

            w_predictions[well][2][fov] = fov_predictions[fov]

        predictions = collections.OrderedDict()
        for well in sorted(w_predictions.keys()):
            w = w_predictions[well]
            predictions[well] = {
                    'prediction': np.divide(w[0], w[1]).tolist(),
                    'fovs': w[2]
                    }
        return predictions

    def save(self, path):
        """
        Saves the localizer to disk.
        Args:
            path: The path to which the localizer should be saved.
        """
        if self._mean is None:
            mean = None
            sigma = None
        else:
            mean = list(self._mean)
            sigma = list(self._sigma)

        data = {
                'zscore': self._zscore,
                'mean': mean,
                'sigma': sigma,
                'classes': self._classes,
               }

        with zipfile.ZipFile(path, 'w') as zip_:
            zip_.writestr('data', json.dumps(data))
            with tempfile.TemporaryDirectory() as tmpdir:
                model = os.path.join(tmpdir, 'model')
                self.dnn.save(model)

                # Have to save all checkpoint files of tensorflow
                for dir_, _, files in os.walk(tmpdir):
                    for f in files:
                        fn = os.path.join(dir_, f)
                        zip_.write(fn, os.path.join('model', f))

    def load(self, path):
        """
        Loads a localizer from disk.
        Args:
            path: The path from which a localizer should be loaded.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path, 'r') as zip_:
                zip_.extractall(path=tmpdir)

            data = os.path.join(tmpdir, 'data')
            with open(data) as f:
                data = json.load(f)
            self._mean = np.asarray(data['mean'])
            self._sigma = np.asarray(data['sigma'])
            self._zscore = data['zscore']
            if 'classes' in data:
                self._classes = data['classes']
            else:
                self._classes = None

            model = os.path.join(tmpdir, 'model', 'model')
            self.dnn = dnn.DNN(model_file=model)
