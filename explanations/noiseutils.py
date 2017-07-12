import os

import pandas as pd
import yaml

from explanations.config import Config

DATASETS = ['beeradvocate', 'yelp', 'tripadvisor']

STRENGTH_FNS = ['bwhi', 'bwsdui', 'bwui', 'bwsd', 'bwsdhi', 'default', 'pc', 'pchi', 'pcui']

DATASET_ALIASES = {'beeradvocate': 'ba', 'tripadvisor': 'ta', 'yelp': 'yp'}

STRENGTH_FN_ALIASES = {
    'default': 'better/worse, no importance',
    'bwhi': 'better/worse, hotel importance',
    'bwui': 'better/worse, user importance',
    'bwsd': 'relative better/worse, no importance',
    'bwsdhi': 'relative better/worse, hotel importance',
    'bwsdui': 'relative better/worse, user importance',
    'pc': 'pros/cons, no importance',
    'pchi': 'pros/cons, hotel importance',
    'pcui': 'pros/cons, user importance',
}


def get_description(dataset, strength_fn):
    args = dict(strength_fn=STRENGTH_FN_ALIASES[strength_fn.lower()],
                dataset=dataset)

    result = '{dataset}, {strength_fn}'.format(**args)
    result = ' '.join([s.strip() for s in result.lower().split(' ') if len(s.strip()) > 1])
    return result.capitalize()


def get_default_config_dict():
    _config_dict = {'items_csv_file': 'data/{dataset}/items.csv',
                    'extractions_csv_file': 'data/{dataset}/extractions.csv',
                    'recommendations_csv_file': 'data/{dataset}/recommendations.csv',
                    'recommendation_size': 9,
                    'sentiment_threshold': 0.7,
                    'compelling_threshold': 0.5,
                    'min_session_length': 10,
                    'strength_function': '{strength_fn}',
                    'es_port': 9200,
                    'es_host': 'http://localhost:9200/',
                    'redis_port': 6379,
                    'redis_swap': './redis.swap',
                    'mongo_port': 28200,
                    'logstash_dir': './',
                    'session_index': '{dataset_abbrv}-{strength_fn}-sessions',
                    'explanation_index': '{dataset_abbrv}-{strength_fn}-explanations',
                    'user_index': '{dataset_abbrv}-{strength_fn}-users',
                    'item_index': '{dataset_abbrv}-{strength_fn}-items',
                    'rec_index': '{dataset_abbrv}-{strength_fn}-rec',
                    'item_ids_csv_file': 'data/{dataset}/item_ids.csv',
                    'user_ids_csv_file': 'data/{dataset}/user_ids.csv',
                    'output_dir': 'outputs/data/{dataset}/{strength_fn}/'}
    return _config_dict


def get_short_name(description):
    aliases = {
        'beeradvocate': 'ba',
        'tripadvisor': 'ta',
        'yelp': 'yp',
        ',': '',
        'relative better/worse': 'rbw',
        'hotel importance': 'himp',
        'user importance': 'uimp',
        'no importance': 'nimp',
        'better/worse': 'bw',
        'pros/cons': 'pc',
        'better': 'b', 'worse': 'w', 'with': '', 'using': '',
        'ratings': '',
        'b/w': 'bw'
    }
    description = description.lower()

    for word, alias in aliases.items():
        for word, alias in aliases.items():
            description = description.replace(word, alias)

    description = '-'.join([s.strip() for s in description.lower().split(' ') if len(s.strip()) >= 1])
    description = description.replace('relative-bw', 'rbw')

    return '{}'.format(description)


def gen_configs():
    config_dicts = []

    config_dict = get_default_config_dict()

    for dataset in DATASETS:
        config_dict['items_csv_file'] = 'data/{}/items.csv'.format(dataset)
        config_dict['extractions_csv_file'] = 'data/{}/extractions.csv'.format(dataset)
        config_dict['recommendations_csv_file'] = 'data/{}/recommendations.csv'.format(dataset)
        config_dict['item_ids_csv_file'] = 'data/{}/item_ids.csv'.format(dataset)
        config_dict['user_ids_csv_file'] = 'data/{}/user_ids.csv'.format(dataset)

        for idx, strength_fn in enumerate(STRENGTH_FNS):

            params = dict(dataset_abbrv=DATASET_ALIASES[dataset.lower()], strength_fn=strength_fn, dataset=dataset)
            keys = ['session_index', 'explanation_index', 'user_index', 'item_index', 'rec_index',
                    'output_dir', 'strength_function']

            for key in keys:
                config_dict[key] = get_default_config_dict()[key].format(**params)

            config_dict['description'] = get_description(dataset=dataset, strength_fn=strength_fn)

            short_name = get_short_name(config_dict['description'])
            config_dict['short_name'] = short_name
            config_dict['explanation_index'] = short_name + '-explanations'
            config_dict['item_index'] = short_name + '-items'
            config_dict['rec_index'] = short_name + '-rec'
            config_dict['session_index'] = short_name + '-sessions'
            config_dict['user_index'] = short_name + '-users'

            config_dicts.append(Config(config_dict=config_dict).__dict__)

    return pd.DataFrame(config_dicts)


def create_config_files(output_dir='./config_files/', exist_ok=True):
    os.makedirs(os.path.realpath(output_dir), exist_ok=exist_ok)
    df_configs = gen_configs()
    file_paths = []

    for idx, row in df_configs.iterrows():
        file_path = os.path.realpath(os.path.join(output_dir, '{}.yml'.format(row['short_name'])))
        with open(file_path, 'w') as f:
            yaml.dump(row.to_dict(), f, default_flow_style=False)

        file_paths.append(file_path)
    return file_paths
