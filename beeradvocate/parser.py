import logging
import os
import re

import numpy as np
import pandas as pd
from IPython.display import display

from explanations.opinionminer import OpinionMiner

__all__ = ['load_reviews', 'fix_column_names', 'fix_datatypes',
           'fix_missing_data', 'split_dataframe', 'preprocess', 'concat_reviews', 'save_reviews_in_mongodb']

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

URI_IN = 'mongodb://127.0.0.1:27017'
URI_OUT = 'mongodb://127.0.0.1:27017'
DB_IN = 'beeradvocate'
DB_OUT = 'beeradvocate'
COLL_IN = 'beeradvocate'
COLL_OUT = 'reviews_beeradvocate'
FIND_QUERY = {}
N_JOBS = 6


TABOO_WORDS = frozenset(['...', '\t'])


def clean(txt):
    """Removes tabs and excessive spaces from a given text"""
    try:
        txt = re.sub('\s+', ' ', txt)
        words = [w for w in txt.split(' ') if w not in TABOO_WORDS]
        return ' '.join(words)
    except AttributeError as e:
        print(e)
        print(txt)
        return txt


def concat_reviews(df_reviews):
    """Concatenate all the reviews for each beer in the reviews DataFrame.

    Args:
        df_reviews (pd.DataFrame): The DataFrame of reviews.

    Returns
        iterator: Iterator of tuples where the first item is the beer ID, and the second is all the reviews of the beer
    """
    assert 'beer_id' in df_reviews.columns
    assert 'review_text' in df_reviews.columns

    for idx, (beer_id, df) in enumerate(df_reviews.groupby('beer_id')):
        yield (beer_id, ' '.join(df.review_text.tolist()))


def preprocess(file_path: str, out_dir=None, verbose=False):
    """Pre-processes BeerAdvocate reviews from source text file.

    Args:
        file_path (str): The source BeerAdvocate text file.
        out_dir (str, optional): Directory to output CSV files for reviews and beers.

    Returns:
        dict: Contains DataFrames of beers and reviews with different keys.
    """

    df_beers = load_reviews(file_path)
    df_beers = fix_column_names(df_beers)
    df_beers = fix_datatypes(df_beers)
    df_beers, missing_dict = fix_missing_data(df_beers)

    if verbose:
        display(df_beers.describe().T)
        print('# Users: {:,}'.format(df_beers.user_id.nunique()))
        print('# Beers: {:,}'.format(df_beers.beer_id.nunique()))

    df_beers, df_reviews = split_dataframe(df_beers)
    df_reviews['review_year'] = df_reviews.review_time.dt.year
    df_reviews['review_text'] = df_reviews.review_text.apply(clean)

    if out_dir and os.path.isdir(out_dir):
        logger.info('Saving data files to {path}'.format(path=out_dir))
        df_beers.to_csv(os.path.join(out_dir, 'beers.csv'), index=False)
        df_reviews.to_csv(os.path.join(out_dir, 'reviews.csv'), index=False)

    logger.info('Finished')
    return dict(beers=df_beers, reviews=df_reviews)


def load_reviews(path, exclude_time=False, exclude_text=False):
    """Loads BeerAdvocate reviews into a DataFrame from source text file."""
    logger.info('Loading reviews from {path}'.format(path=path))

    def _read():
        """Reads the BeerAdvocate reviews"""
        with open(path) as f:
            record = {}
            for line in f.readlines():
                if exclude_time and 'time' in line:
                    continue

                if exclude_text and 'text' in line:
                    continue

                line = line.strip()
                if len(line) == 0:
                    yield record
                    record = {}
                else:
                    index = line.index(':')
                    feature = line[:index].replace('/', '_').lower()
                    value = line[index + 1:].strip().lower()
                    record[feature] = value

    records = _read()
    return pd.DataFrame.from_records(records)


def fix_column_name(column_name):
    column_name = column_name.replace('beer_beerid', 'beer_id')
    column_name = column_name.replace('beer_brewerid', 'brewer_id')
    column_name = column_name.replace('review_profilename', 'user_id')
    return column_name


def fix_column_names(df):
    logger.info('Renaming columns...')
    df.columns = [fix_column_name(c) for c in df.columns]
    return df


def fix_datatypes(df):
    logger.info('Fixing data types...')

    def fix_beer_abv(abv):
        return float('nan') if str(abv).strip() == '' else float(abv)

    df.beer_abv = df.beer_abv.apply(fix_beer_abv)

    if 'beer_id' in df.columns:
        logger.info('Coercing data type of beer_id to str...')
        df['beer_id'] = df.beer_id.astype(str)
    else:
        logger.warning('Failed to coerce data type of beer_id to str')

    for col in df.columns:
        if col.startswith('review_'):
            if col == 'review_time':
                from datetime import datetime
                df[col] = df[col].apply(lambda x: datetime.fromtimestamp(int(x)))
            elif col == 'review_text':
                pass
            else:
                df[col] = df[col].astype(float)

    return df


def fix_missing_data(df):
    logger.info('Fixing missing data...')
    missing_dict = {}

    for col in df.columns:
        n_missing = len(df[df[col].isnull()])

        if n_missing > 0:
            missing_dict[col] = n_missing

    df = df[~df.user_id.isnull()].copy()
    df['beer_name'] = df.beer_name.fillna('(no name)')
    df['beer_style'] = df.beer_style.apply(lambda s: s.replace(' / ', '/'))

    return df, missing_dict


def split_dataframe(df):
    """
    Splits combined DataFrame of beers and reviews into separate DataFrame

    Returns two DataFrames: first with beer data, and second with review data.
    """
    logger.info('Extracting beer information from reviews...')
    cols_reviews = [c for c in df.columns if c.startswith('review_')
                    or c in ('user_id', 'beer_id')]
    cols_beers = [c for c in df.columns if c.startswith('beer_')]

    if 'review_time' in df.columns:
        df_reviews = df[cols_reviews].sort_values('review_time')
    else:
        df_reviews = df[cols_reviews]

    # create review_id column
    df_reviews['review_id'] = np.arange(len(df_reviews)) + 1
    df_reviews['review_id'] = df_reviews.review_id.apply(lambda x: 'r{}'.format(x))

    # remove duplicate beer data
    df_beers = df[cols_beers].drop_duplicates()

    # add average rating to beer dataframe
    df_avg = df_reviews.groupby('beer_id').review_overall.mean().reset_index()
    df_avg.rename(columns={'review_overall': 'average_rating'}, inplace=True)
    df_beers = pd.merge(df_beers, df_avg, on='beer_id', how='inner')

    return df_beers, df_reviews

