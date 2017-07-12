"""Logic for clustering features mined from BeerAdvocate reviews.

Copyright (c) 2010, Khalil Muhammad
"""
import logging
import math
from collections import OrderedDict
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

__all__ = ['FeatureMapper', 'FeatureInfo', 'centroids_from_categories', 'get_sentence_bow', 'cluster_features']

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

FEATURE_CATEGORIES = ['appearance', 'taste', 'aroma', 'palate']
FEATURE_AMENITY_MAP = {
    'appearance': 'appearance',
    'taste': 'taste',
    'aroma': 'aroma',
    'palate': 'palate'
}

FeatureInfo = namedtuple('FeatureInfo', ['amenity', 'base_feature', 'expanded_features'])


class FeatureMapper(object):
    def __init__(self, df_extractions: pd.DataFrame, feature_categories: list = None, feature_amenity_map: dict = None):
        """Constructor

        Args:
            df_extractions (pd.DataFrame): Opinions extracted from reviews
            feature_categories (list): List of feature categories
            feature_amenity_map (dict): Maps features to amenities
        """
        self.clusters = None
        self.tdm = None
        self.kmeans = None
        self.df_extractions = df_extractions
        self.feature_categories = feature_categories if feature_categories else FEATURE_CATEGORIES
        self.feature_amenity_map = feature_amenity_map if feature_amenity_map else FEATURE_AMENITY_MAP
        self.feature_map = dict()  # The feature_map is a dictionary that maps feature indices to features.

    def _create_tdm(self):
        """Create a term-document matrix"""
        feature_sentences = list(self.get_sentence_bow())
        vectorizer = TfidfVectorizer()
        self.tdm = vectorizer.fit_transform(feature_sentences)
        return self.tdm

    def cluster_features(self):
        """Cluster features into given categories.

        Args:
        Returns:
            dict: Map of category features to other related features.
        """
        logger.info('Creating term-document matrix...')
        self._create_tdm()
        init_centroids = self.centroids_from_categories()

        # Cluster the features using specific centroids.
        logger.info('Clustering features...')
        self.kmeans = KMeans(init=init_centroids, n_init=1, max_iter=1, n_clusters=len(self.feature_categories))
        self.clusters = self.kmeans.fit_predict(self.tdm)

        # The feature vector maps key features (categories) to other features that occur in the same cluster.
        logger.info('Converting clusters to feature vectors...')
        feature_vectors = self.clusters_to_feature_vectors(category_features=list(self.feature_amenity_map.keys()))

        return feature_vectors

    def get_doc_name_and_dist(self, cluster_idx, features):
        """Get the document name closest to a cluster's centriod

        Args:
            cluster_idx (int): The index of the cluster.
            features:
        """
        # Get all documents for cluster c_idx
        td_vectors = self.tdm[np.where(self.clusters == cluster_idx)]
        features = np.array(features)
        doc_names_ = features[np.where(self.clusters == cluster_idx)]

        # Compute Euclidean distances of all documents
        # to the centroid of the cluster.
        doc_dist = dict()

        for i in range(td_vectors.shape[0]):
            # Compute distance between document i and centroid of the cluster.
            doc_dist[doc_names_[i]] = euclidean(td_vectors[i].A[0], self.kmeans.cluster_centers_[cluster_idx])

        # Sort the document distances in descending order of their distances.
        doc_dist_ordered = OrderedDict(sorted(doc_dist.items(), key=lambda t: t[1]))
        return doc_dist_ordered

    def clusters_to_dict(self, features):
        _clusters = {}
        for idx in range(0, self.kmeans.n_clusters):
            # _cluster = self.get_doc_name_and_dist(cluster_idx=idx, features=features).items()
            _cluster = self.get_doc_name_and_dist(cluster_idx=idx, features=list(self.feature_map.values())).items()
            _cluster = OrderedDict(sorted(_cluster, key=lambda x: x[1], reverse=True))
            _clusters[idx] = _cluster
        return _clusters

    def clusters_to_feature_vectors(self, category_features=None):
        """Converts clusters into a dictionary of categories and features

        Args:
            category_features (list): Features corresponding to categories.

        Returns:
            dict: Map of category features to a list of similar/related features.
        """
        feature_vectors = {}
        # _clusters = self.clusters_to_dict(features=category_features)
        _clusters = self.clusters_to_dict(features=list(self.feature_map.values()))

        for feature in sorted(category_features):
            for cluster_idx, clustered_features in _clusters.items():
                _clustered_features = dict(clustered_features)
                if feature in _clustered_features.keys():
                    distance = _clustered_features[feature]  # distance from cluster centre.

                    for _feature, _distance in _clustered_features.items():
                        if _feature != feature:
                            _clustered_features[_feature] = math.fabs(_distance - distance)

                    _clustered_features_sorted = OrderedDict(sorted(_clustered_features.items(), key=lambda x: x[1]))

                    feature_vectors[feature] = [f for f in _clustered_features_sorted.keys()
                                                if f.lower() != feature.lower()]
        return feature_vectors

    def centroids_from_categories(self):
        """Get rows in the term-document matrix that corresponding to the given feature categories.

        Returns:
            np.ndarray: List of rows corresponding to feature categories in the input term-document matrix.
        """
        # From the term-document matrix, get rows corresponding to the given feature categories.
        # This will be used to initialize the KMeans clustering algorithm.
        feature_map_inv = {v: k for k, v in self.feature_map.items()}
        rows = list()
        for feature in self.feature_categories:
            feature_idx = feature_map_inv[feature]
            rows.extend(self.tdm[feature_idx].todense().tolist())

        rows = np.array(rows)
        return rows

    def get_sentence_bow(self):
        """Yields sentences linked with features in a DataFrame of extractions.

        Yields:
            iterator: Each entry is a string corresponding to all sentences linked with a given feature.
        """
        for idx, (feature, df) in enumerate(self.df_extractions.groupby('feature')):
            self.feature_map[idx] = feature
            yield ' '.join(df.sentence_str.tolist())
