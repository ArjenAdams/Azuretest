import pandas as pd
from sklearn.metrics import pairwise_distances


def get_similarity_list(data, target_index, method='jaccard',
                        sort='descending', n_items=-1, unique_only=False,
                        exclude_self=True):
    df = pd.get_dummies(data)
    distances = pairwise_distances(df, metric=method)
    similarities = 1 - distances
    similarities = pd.DataFrame(similarities, index=data.index, columns=data.index)
    similarities_list = similarities[target_index].sort_values(ascending=sort == 'ascending')
    if exclude_self:
        similarities_list = similarities_list[similarities_list.index != target_index]
    if unique_only:
        similarities_list = similarities_list[similarities_list == 1.0]
    if n_items > 0:
        similarities_list = similarities_list[:n_items]
    return similarities_list
