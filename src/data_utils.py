import numpy as np
import random
import itertools
import tempfile
import pickle
from proteinlocalizer.utils.iter_utils import repeatable_generator


@repeatable_generator
def __read_pickle_file(f):
    f.seek(0)
    while True:
        try:
            yield pickle.load(f)
        except EOFError:
            break


def external_shuffle(iterable):
    """
    Creates a temporary file with the data in the iterable.
    The data must be pickleable for the function to work.

    Note that this method is inefficient! If you work with data that can fit
    into memory, an ordinary shuffle is MUCH better to use.

    Params:
        iterable: The iterable from which to shuffle the data.
    Returns:
        An generator which yields data from the iterable in random order.
        The generator can be used multiple times, but the order stays the same
        each time.
    """
    tmp_storage = tempfile.TemporaryFile()
    lines = []
    for line in iterable:
        lines.append(tmp_storage.tell())
        pickle.dump(line, tmp_storage)

    random.shuffle(lines)
    shuffled_file = tempfile.TemporaryFile()
    for i in lines:
        tmp_storage.seek(i)
        item = pickle.load(tmp_storage)
        pickle.dump(item, shuffled_file)

    shuffled_file.seek(0)
    return __read_pickle_file(shuffled_file)


def kfold(data, k):
    """
    Performs kfold split on the dataset and returns the pieces in succession.
    If k divides len(data) perfectly, the pieces will always be the same size,
    otherwise it may differ in one instance.

    Does NOT shuffle the list before splitting.

    Args:
        data: An iterable of data to be validated.

        k: The number of splits to perform.

    Returns:
        A generator which will yield tuples of data (train, test).
        test will be approximately 1 kth the size of data while train will
        include all other data from the set.
        All possible combinations of the k pieces will be yielded from the
        generator.
    """
    data = np.asarray(data)
    data_split = np.array_split(data, k)
    for i in range(k):
        test = data_split[i]
        train = [data_split[j] for j in range(k) if j != i]
        train = np.concatenate(train)
        yield train, test


def external_kfold(iterable, k=5):
    """
    Externally performs k-fold cross validation.
    Requires the data generated by iterable to be pickleable.

    Args:
        k: The number of folds to perform in the cross validation.

    Returns:
        A list of tuples containing generators [(tr0, te0), (tr1, te1)...].
        Each element in the list corresponds to one fold, where the first tuple
        element will generate training data and the second testing data.
    """
    tmpfiles = []
    for _ in range(k):
        tmpfiles.append(tempfile.NamedTemporaryFile())

    for (i, g) in enumerate(iterable):
        i %= k
        pickle.dump(g, tmpfiles[i])

    train_tests = []
    for k_i in range(k):
        tr_files = [open(tmpfiles[x].name, 'rb') for x in range(k) if x != k_i]
        train_reads = [__read_pickle_file(f) for f in tr_files]

        train_full = itertools.chain(*train_reads)

        test_file = open(tmpfiles[k_i].name, 'rb')
        test_read = __read_pickle_file(test_file)

        train_tests.append((train_full, test_read))

    return train_tests


def external_zscore(iterable, mean=None, stddev=None):
    """
    Computes the zscore of the data in the iterable on disk rather than in
    memory.
    See zscore for the output format.
    """
    tmp_storage = tempfile.TemporaryFile()
    normalized = tempfile.TemporaryFile()
    linecount = 0
    calc_mean = True if mean is None else False
    for line in iterable:
        line = np.asarray(line)
        if mean is None:
            mean = np.zeros(len(line))
            s1 = np.zeros(len(line))
            s2 = np.zeros(len(line))

        linecount += 1
        pickle.dump(line, tmp_storage)
        if calc_mean:
            mean += line
            s1 += line
            s2 += line ** 2

    if not linecount:
        raise ValueError('Cannot zscore an empty vector')

    if calc_mean:
        stddev = np.sqrt(linecount*s2 - s1**2)/linecount
        mean /= linecount

    tmp_storage.seek(0)
    while True:
        try:
            line = pickle.load(tmp_storage)
            line = np.asarray(line)
            norm = (line-mean)/stddev
            pickle.dump(norm, normalized)
        except EOFError:
            break
    normalized.seek(0)
    return __read_pickle_file(normalized), mean, stddev


def external_split(iterable, split):
    """
    Splits the data into two parts externally and returns the parts seperately.
    Takes the first `split` items of the as the separate items.
    The data generated by iterable needs to be pickleable for the function to
    be usable.

    Returns:
        Two generators (data_gen, split_gen). data_gen will yield all data from
        the iterable except for the first `split` items while split_gen will
        generate the other items.
    """
    if not isinstance(split, int):
        raise TypeError('split needs to be an integer')

    split_file = tempfile.TemporaryFile()
    data_file = tempfile.TemporaryFile()
    for g in iterable:
        if split > 0:
            pickle.dump(g, split_file)
            split -= 1
        else:
            pickle.dump(g, data_file)
    split_file.seek(0)
    split_gen = __read_pickle_file(split_file)
    data_file.seek(0)
    data_gen = __read_pickle_file(data_file)
    return (data_gen, split_gen)


def shuffle_lists(*lists):
    """
    Shuffles several lists simultaneously.

    The length of all the lists has to be the same.

    Params:
        lists   :   A list of lists to be shuffled
    Returns:
        The list of shuffled lists
    """

    shuffled_lists = [[] for l in lists]
    indexes = range(len(lists[0]))
    random.shuffle(indexes)

    for i in indexes:
        for shuffled, real in itertools.izip(shuffled_lists, lists):
            shuffled.append(real[i])

    return shuffled_lists


def zscore(data, mean=None, stddev=None):
    """
    Calculates the zscore of the data based on the supplied mean and standard
    deviation.

    The zscore formula is:
        z = (data-mean)/stddev
    Args:
        data:   The data to be zscored.
        mean:   The mean to use in the zscore formula.
                If None, the mean of data is used.
        stddev: The standard deviation to use in the zscore formula.
                If None, the standard deviation of data is used.
    Returns:
        A tuple on the format (z, mean, stddev) where z is the zscored data,
        mean is the mean used in the calculation, and stddev is the standard
        deviation used in the formula.
    """
    if mean is None:
        mean = np.mean(data, axis=0)
    if stddev is None:
        stddev = np.std(data, axis=0)
    data = np.asarray(data)
    mean = np.asarray(mean)
    stddev = np.asarray(stddev)
    z = (data-mean)/stddev
    return z, mean, stddev


def exact_match(reals, predicts):
    """
    Calculates the exact match score of the predictions on the real labels. The
    exact match score is defined as

    sum(1 if s==t, otherwise 0 for each s in S and each t in T))/len(T)
    where T is the set of real labels and S the set of predictions.

    Parameters:
        reals: A list of binary vectors.
               The list should consist of the target vectors.
        predicts: A list of binary vectors.
                  The list should consist of the prediction vectors.
    Returns:
        The exact match score of the predictions on the labels.
    """
    accs = list(map(lambda x: x[0] == x[1], zip(reals, predicts)))
    if not len(accs):
        raise ValueError('The lists need to contain values')
    return np.mean(accs)


def hamming_score(reals, predicts):
    """
    Calculates the Hamming Score as defined by Godbole and Sarawagi in
    Discriminative Methods for Multi-labeled Classification.

    Let reals=T, and predicts=S.
    The Hamming Score is calculated as
    1/len(S) * |intersection(T,S)|/|union(T,S)|

    Parameters:
        reals   :   A list of binary vectors.
                    The list should consist of the target vectors.

        predicts:   A list of binary vectors.
                    The list should consist of the prediction vectors.
    Returns:
        The Hamming score of the predictions on the real labels.
    """
    numerator = 0
    denominator = 0
    for (r, p) in zip(reals, predicts):
        if len(r) != len(p):
            raise ValueError('Array lengths do not agree')

        true = set(np.where(r)[0])
        pred = set(np.where(p)[0])

        intersection = true.intersection(pred)
        union = true.union(pred)
        numerator += len(intersection)
        denominator += len(union)
    return numerator/denominator


def precision_recall(reals, predicts, instance='total'):
    """
    Calculates the total precision and recall.
    Args:
        reals:  A list of binary vectors.
                The list should consist of the actual labels.
        predicts:   A list of binary vectors.
                    The list should consist of the predicted labels.
        instance: Determines on what level the precision and recall should be
                  calculated. The current possible options are:
                    total - Calculate overall precision and recall
                    class - Calculate precision and recall per class

    Returns:
        The precision and recall, as a tuple, of the
        predictions made on the dataset.
        The tuple can consist of lists should it be fitting for the instance
        type.

        In the total instance:
            Returns a tuple consisting of 2 values, precision and recall
        In the class instance:
            Returns a tuple of two lists. The first list contains the precision
            per class, the second the recall.
    """

    if instance == 'total':
        axis = None
    elif instance == 'class':
        axis = 0
    else:
        raise ValueError('Not a valid instance type')

    reals = np.asarray(reals)
    predicts = np.asarray(predicts)

    truepos = np.logical_and(reals, predicts)
    false = reals - predicts
    falseneg = false > 0
    falsepos = false < 0

    truepos = np.sum(truepos, axis=axis)
    falseneg = np.sum(falseneg, axis=axis)
    falsepos = np.sum(falsepos, axis=axis)

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = truepos/(truepos + falsepos)
        recall = truepos/(truepos + falseneg)
        if not np.isscalar(precision):
            precision[~np.isfinite(precision)] = 0
            recall[~np.isfinite(recall)] = 0
    return precision, recall


def accuracy(reals, predicts, instance='total'):
    """
    Calculates the accuracy of the predictions on the real labels.
    The accuracy here is defined to be the (TP + TN)/(TP + TN + FP + FN) where
    TP=True Positive, TN=True Negative, FP=False Positive and FN=False Negative

    Args:
        reals: A list of binary vectors.
               The list should contain the actual labels.
        predicts: A list of binary vectors.
                  The list should contain the predicted labels.
        instance: A string determining what level the accuracy should be
                  calculated on. The current options are:
                  total - Calculate the overall accuracy
                  class - Calculate the accuracy per class

    Returns:
        The accuracy of the predictions.
        The format of this depends on what instance was used.

        In the total instance:
            Returns a real valued number
        In the class instance:
            Returns a real valued number per class as a list.
    """
    if instance == 'total':
        axis = None
    elif instance == 'class':
        axis = 0
    else:
        raise ValueError('Not a valid instance type')
    falses = np.logical_xor(reals, predicts)
    truths = np.logical_not(falses)

    truths = np.sum(truths, axis=axis)
    falses = np.sum(falses, axis=axis)
    return truths/(falses + truths)