import itertools
import random

import numpy as np


def get_all_possible_matchings(iterable1, iterable2):
    # type: (list, list) -> list
    """
    a = ["foo", "melon"]
    b = [True, False]
    [("foo", True), ("foo", False), ("melon", True), ("melon", False)]
    :param iterable1: iterable object
    :param iterable2: iterable object
    :return: list of all possible matchings between the 2 iterables
    """
    return list(itertools.product(iterable1, iterable2))


def get_max_argmax(to_value_func, items_list):
    # type: (callable, list) -> (float, object)
    """
    Returns (maximum value, item that gets the maximum value)
    :param to_value_func: receives an item from $items_list and returns a float value
    :param items_list: list of items
    :return: (maximum value, item that gets the maximum value)
    """
    values_list = map(to_value_func, items_list)
    max_val = max(values_list)
    arg_max = items_list[values_list.index(max_val)]  # get the right item from items_list
    return max_val, arg_max


def flatten_lists(listOfLists):
    """
    takes a list of lists and returns a concatenated list of all of them
    """
    return [item for sublist in listOfLists for item in sublist]


def lists_dot(l1, l2):
    """
    applies dot product between 2 numeric lists
    """
    res = 0.0
    for i in range(len(l1)):
        res += l1[i] + l2[i]

    return res
    # return map(lambda (x, y): x * y, zip(l1, l2))


def concat_np_arrays(np_arrays):
    res = []
    for arr in np_arrays:
        res += arr.tolist()
    return np.array(res)


def unique_list(l):
    # type: (list) -> list
    """
    returns a list with only unique items from $l
    """
    return list(set(l))


def create_comp(comp_sents, tagsents, file_name):
    i = 0
    with open(file_name, 'w') as file:
        for (sentence, tags) in zip(comp_sents, tagsents):
            tagged = map(lambda (w, t): w + '_' + t, zip(sentence, tags))
            i += 1
            if i == len(comp_sents):
                file.write(' '.join(tagged))
            else:
                file.write(' '.join(tagged) + '\n')


def split_list(file_lines, separator):
    # type: (list, str) -> list
    """
    separates lists according to a specific separator
    """
    return [list(y) for x, y in itertools.groupby(file_lines, lambda z: z == separator) if not x]


def weighted_random_choice(weights):
    weights_sum = sum(weights)
    if weights_sum == 0:
        # print "all scores are 0"
        weights = [1.0] * len(weights) # map(lambda w: w + 1.0, weights)
        weights_sum = sum(weights)
    rnd = random.random() * weights_sum
    for i, w in enumerate(weights):
        rnd -= w
        if rnd < 0:
            return i


def weighted_random_choice_bulk(weights, count, allow_dups=False):
    abs_min_weight = abs(min(min(weights), 0))
    weights = map(lambda w: w+abs_min_weight+1, weights)
    if not allow_dups:
        chosen_idxs = set()
        assert count <= len(filter(lambda x: x!=0, weights)), \
            "If no duplication allowed, can't return more items than in $weights"
        while len(chosen_idxs) < count:
            chosen_idxs.add(weighted_random_choice(
                map(lambda idx: weights[idx] if idx not in chosen_idxs else 0, range(len(weights)))))
    else:
        chosen_idxs = []
        for _ in range(count):
            chosen_idxs.append(weighted_random_choice(weights))

    assert len(chosen_idxs) == count
    return list(chosen_idxs)


def random_choice_bulk(lst, count, allow_dups=False):
    return weighted_random_choice_bulk([1] * len(lst), count, allow_dups)  # same weight for all = random


def lists_minlen(*args):
    """
    Returns the minimal length of all the parameters in $args
    """
    return min(map(len, args))


def lists_maxlen(*args):
    """
    Returns the maximal length of all the parameters in $args
    """
    return max(map(len, args))