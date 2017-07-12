from __future__ import division, print_function

import logging
import os
import time
from collections import deque
from itertools import islice, count

import numpy as np
import pandas as pd
from IPython.display import display


def walk_path(path: str, file_extension: str):
    """Walks a path and returns files with a given extension.

    Args:
        path (str): The path to walk
        file_extension (str): The extension of the files to return

    Yields:
        List of file paths that matches the given extension
    """
    for (root, directories, files) in os.walk(path):
        for filename in files:
            if filename.endswith(file_extension):
                file_path = os.path.join(os.path.realpath(root), filename)
                yield file_path


def unpack(x):
    """Unpacks elements that are deserialized to lists."""
    if isinstance(x, list) and len(x) == 1:
        return x[0]
    else:
        return x


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        tc_s = time.clock()
        result = method(*args, **kw)
        te = time.time()
        tc_e = time.clock()

        print("{} time.time({}), time.clock({}) sec".format(method.__name__, te - ts, tc_e - tc_s))
        return result

    return timed


def check_session_sizes(df_explanations, expected_session_size=10):
    """Checks that all sessions are of expected length and complete."""
    logging.info('Checking consistency of session sizes in explanations...')
    uniques = df_explanations.groupby('session_id').size().unique()
    assert uniques[0] == expected_session_size, 'Unexpected session size'
    assert len(uniques) == 1, 'Sessions are of different sizes'


def scale(x, x_min=0, x_max=5, y_min=-1, y_max=1):
    """Scales a value x from one range [x_min,x_max] to another [y_min,y_max]"""
    assert x_min < x_max and y_min < y_max
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min


def drop_columns_if_exists(df, columns, inplace=False):
    """Drops columns from a DataFrame"""
    for c in columns:
        if c in df.columns:
            df.drop(c, axis=1, inplace=inplace)

    if not inplace:
        return df


def describe_explanations(df_main):
    print('# Explanations: {:,}'.format(df_main.explanation_id.nunique()))
    print('# Seed Explanations: {:,}'.format(len(df_main.query('is_seed == True'))))
    print('# Sessions: {:,}'.format(df_main.session_id.nunique()))
    print('# Items: {:,}'.format(df_main.item_id.nunique()))
    print('# Seed Items: {:,}'.format(df_main.seed_item_id.nunique()))
    df_main['user_id'] = df_main.session_id.apply(lambda x: x.split('#')[0])
    print('# Users: {:,}'.format(df_main.user_id.nunique()))
    percentage = int(np.round(100 * len(df_main.query('is_comp == True')) / len(df_main)))
    print('% of explanations that are compelling: {}%'.format(percentage))
    display(df_main.describe().T[['mean', 'std', 'min', '50%', 'max']])


def validate_item_profiles(df_item_profiles, n_recommendations=10):
    """Checks that all recommendations in a item profile are of the expected size,
    and have valid similarity values.
    """
    logging.info('Validating item profiles...')
    assert df_item_profiles.related_items_sims.apply(len).unique()[0] == n_recommendations
    assert df_item_profiles.related_items.apply(len).unique()[0] == n_recommendations
    result = df_item_profiles.related_items_sims.apply(lambda x: all(~np.isnan(x)))
    assert result.nunique() == 1, 'Some related_items have NaN values in them'
    assert result.values[0] == True

    # df_item_profiles.related_items has to be a list of string values.
    assert isinstance(df_item_profiles.related_items.values[0], list)

    # df_item_profiles.related_items_sims has to be an np.ndarray
    value = df_item_profiles.related_items_sims.values[0]
    assert isinstance(value, np.ndarray), 'related_items_sims is of type {}'.format(type(value))
    return


def fake_personalized_similarities(df_recommendations, df_user_profiles):
    """
    Converts non-personalized recommendations to personalized ones.
    """
    aliases = {'beer_id': 'item_id', 'recommendations': 'related_items', 'similarities': 'related_items_sims'}
    df_similarities = df_recommendations.rename(columns=aliases)
    df_similarities['personalized_sims'] = df_similarities.related_items_sims

    def _get_user_items():
        for idx, row in df_user_profiles.iterrows():
            if isinstance(row.item_ids, str):
                yield dict(user_id=row.user_id, item_id=row.item_ids)
            else:
                for item_id in row.item_ids:
                    yield dict(user_id=row.user_id, item_id=item_id)

    df_user_item_ids = pd.DataFrame(_get_user_items())

    df_similarities = pd.merge(df_user_item_ids, df_similarities, on='item_id', how='left')
    assert len(df_user_item_ids) == len(df_similarities)
    return df_similarities


def take(n, iterable):
    """Return first n items of the iterable as a list"""
    return list(islice(iterable, n))


def split_every(n, iterable):
    i = iter(iterable)
    piece = list(islice(i, n))
    while piece:
        yield piece
        piece = list(islice(i, n))

    return


def consume(iterator, n):
    """Advance the iterator n-steps ahead. If n is none, consume entirely."""
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)

    return


def count_iter_items(iterable):
    """
    Consume an iterable not reading it into memory; return the number of items.
    http://stackoverflow.com/a/15112059
    """
    counter = count()
    deque(zip(iterable, counter), maxlen=0)  # (consume at C speed)
    return next(counter)


class Singleton(type):
    def __init__(cls, name, bases, dict):
        super(Singleton, cls).__init__(name, bases, dict)
        cls.instance = None

    def __call__(cls, *args, **kw):
        if cls.instance is None:
            cls.instance = super(Singleton, cls).__call__(*args, **kw)

        return cls.instance
