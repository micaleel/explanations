import logging
import os
import re

import numpy as np
import pandas as pd
from IPython.display import display

from explanations.opinionminer import OpinionMiner

__all__ = ['load_reviews', 'fix_column_names', 'fix_datatypes',
           'fix_missing_data', 'split_dataframe', 'preprocess_beeradvocate', 'concat_reviews',
           'save_reviews_in_mongodb']

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
    """Concatenate all the reviews for each item in the reviews DataFrame.

    Args:
        df_reviews (pd.DataFrame): The DataFrame of reviews.

    Returns
        iterator: Iterator of tuples where the first item is the item ID, and the second is all the reviews of the item
    """
    assert 'item_id' in df_reviews.columns
    assert 'review_text' in df_reviews.columns

    logger.info('Removing empty reviews...')
    df_reviews.dropna(subset=['review_text'], inplace=True)
    df_reviews['review_text'] = df_reviews.review_text.astype(str)

    if not isinstance(df_reviews.item_id.values[0], str):
        logger.info('Coercing item IDs to strings...')
        df_reviews['item_id'] = df_reviews.item_id.astype(str)

    for idx, (item_id, df) in enumerate(df_reviews.groupby('item_id')):
        try:
            yield (item_id, ' '.join(df.review_text.tolist()))
        except TypeError as e:
            logger.error(e)
            x = df.review_text.tolist()
            logger.info('item_id: {}, type(item_id): {}'.format(item_id, type(item_id)))
            logger.info('len(df.review_text.tolist()): {}, df.review_text.tolist(): {} '.format(len(x), x))


def preprocess_beeradvocate(file_path: str, out_dir=None, verbose=False):
    """Pre-processes BeerAdvocate reviews from source text file.

    Args:
        file_path (str): The source BeerAdvocate text file.
        out_dir (str, optional): Directory to output CSV files for reviews and items.
        verbose (bool):
    Returns:
        dict: Contains DataFrames of items and reviews with different keys.
    """

    df_items = load_reviews(file_path)
    df_items = fix_column_names(df_items)
    df_items = fix_datatypes(df_items)
    df_items, missing_dict = fix_missing_data(df_items)

    # # Fix IDs, append fist letter of column character to numeric ID field
    # for col in ('review_id', 'item_id', 'beer_id', 'review_profilename', 'user_id'):
    #     if col in df_items.columns:
    #         prefix = 'u' if col == 'review_profilename' else col[0]
    #         df_items[col] = df_items[col].apply(lambda x: '{}{}'.format(prefix, x))

    if verbose:
        display(df_items.describe().T)
        print('# Users: {:,}'.format(df_items.user_id.nunique()))
        print('# Beers: {:,}'.format(df_items.item_id.nunique()))

    df_items, df_reviews = split_dataframe(df_items)
    df_reviews['review_year'] = df_reviews.review_time.dt.year
    df_reviews['review_text'] = df_reviews.review_text.apply(clean)

    if out_dir and os.path.isdir(out_dir):
        logger.info('Saving data files to {path}'.format(path=out_dir))
        df_items.to_csv(os.path.join(out_dir, 'beers.csv'), index=False)
        df_reviews.to_csv(os.path.join(out_dir, 'reviews.csv'), index=False)

    logger.info('Finished')
    df_items.rename(columns={'beer_id': 'item_id'}, inplace=True)
    df_reviews.rename(columns={'beer_id': 'item_id', 'review_profilename': 'user_id'}, inplace=True)
    return dict(items=df_items, reviews=df_reviews)


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
    column_name = column_name.replace('beer_beerid', 'item_id')
    column_name = column_name.replace('beer_brewerid', 'brewer_id')
    column_name = column_name.replace('review_profilename', 'user_id')
    column_name = column_name.replace('beer_name', 'item_name')
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

    if 'item_id' in df.columns:
        logger.info('Coercing data type of item_id to str...')
        df['item_id'] = df.item_id.astype(str)
    else:
        logger.warning('Failed to coerce data type of item_id to str')

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
    df['item_name'] = df.item_name.fillna('(no name)')
    df['beer_style'] = df.beer_style.apply(lambda s: s.replace(' / ', '/'))

    return df, missing_dict


def split_dataframe(df):
    """
    Splits combined DataFrame of beers and reviews into separate DataFrame

    Returns two DataFrames: first with beer data, and second with review data.
    """
    logger.info('Extracting beer information from reviews...')
    cols_reviews = [c for c in df.columns if c.startswith('review_')
                    or c in ('user_id', 'item_id')]
    cols_beers = [c for c in df.columns if c.startswith('beer_') or c in ('beer_id', 'item_id', 'item_name')]

    if 'review_time' in df.columns:
        df_reviews = df[cols_reviews].sort_values('review_time')
    else:
        df_reviews = df[cols_reviews]

    # create review_id column
    df_reviews['review_id'] = np.arange(len(df_reviews)) + 1
    df_reviews['review_id'] = df_reviews.review_id.apply(lambda x: 'r{}'.format(x))

    # remove duplicate beer data
    df_items = df[cols_beers].drop_duplicates()

    # add average rating to beer dataframe
    df_avg = df_reviews.groupby('item_id').review_overall.mean().reset_index()
    df_avg.rename(columns={'review_overall': 'average_rating'}, inplace=True)
    logger.info('df_avg.columns: {}'.format(df_avg.columns.tolist()))
    logger.info('df_items.columns: {}'.format(df_items.columns.tolist()))
    df_items = pd.merge(df_items, df_avg, on='item_id', how='inner')

    return df_items, df_reviews


def save_reviews_in_mongodb(df_reviews, dataset: str, batch_id='main', out_dir=None, site_name: str = None,
                            url: str = None):
    assert dataset in ('yelp', 'beeradvocate', 'tripadvisor')

    site_name = site_name if site_name else '(none)'
    url = url if url else '(none)'

    # convert reviews to format expected by OMF
    logger.info('Converting reviews to format expected by OMF...')
    columns = ['review_id', 'item_id', 'user_id', 'beer_id', 'average_rating', 'review_time', 'review_text',
               'review_overall', 'review_profilename']
    columns = list(set(columns).intersection(df_reviews.columns.tolist()))

    col_aliases = {
        'business_id': 'item_id',
        'beer_id': 'item_id',
        'stars': 'rating',
        'average_rating': 'rating',
        'date': 'rating_date',
        'review_profilename': 'user_id',
        'review_overall': 'rating',
        'review_time': 'rating_date',
    }

    df_reviews = df_reviews[columns].rename(columns=col_aliases)
    logger.info('df_reviews.columns: {}'.format(df_reviews.columns.tolist()))
    df_reviews = df_reviews[df_reviews.review_text.str.strip() != ""]
    df_reviews['url'] = df_reviews.review_id.apply(lambda x: '{}/{}'.format(url, x))
    df_reviews['review_title'] = '(none)'
    df_reviews['batch_id'] = batch_id
    df_reviews['site_name'] = site_name
    df_reviews['item_id'] = df_reviews.item_id.astype(str)

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    # split reviews into smaller chunks and save them to disk.
    review_id_chunks = list(chunks(df_reviews.review_id.unique().tolist(), 10000))

    if out_dir and os.path.isdir(out_dir):
        logger.info('Splitting reviews dataframe into smaller bits to store on disk at {}...'.format(out_dir))

        for idx in range(len(review_id_chunks)):
            dataframe = df_reviews[df_reviews.review_id.isin(review_id_chunks[idx])]
            dataframe.to_csv(os.path.join(out_dir, '{}_{}.csv').format(dataset, idx), index=False)

    miner = OpinionMiner(db_name=dataset, collection_name=dataset)
    db = miner.init_mongodb()
    db['{}_reviews'.format(dataset)].insert_many(df_reviews.to_dict(orient='records'))

    logger.info('Finished')
