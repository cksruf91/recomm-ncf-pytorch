import argparse
import os
import random
from itertools import accumulate
from typing import Tuple, Any

from pandas import DataFrame

from common.loading_functions import loading_brunch, loading_movielens_1m, loading_movielens_10m
from common.utils import DefaultDict
from config import CONFIG


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='10M', choices=['10M', '1M', 'BRUNCH'], help='데이터셋', type=str)
    return parser.parse_args()


def get_negative_samples(train_df, test_df, user_col, item_col, n_sample=99):
    negative_sampled_test = []

    # 샘플링을 위한 아이템들의 누적합
    train_df['item_count'] = 1
    item_counts = train_df.groupby(item_col)['item_count'].sum().reset_index()
    item_counts['cumulate_count'] = [c for c in accumulate(item_counts.item_count)]

    # 샘플링을 위한 변수
    item_list = item_counts[item_col].tolist()
    item_cumulate_count = item_counts['cumulate_count'].tolist()

    # 유저가 이전에 interaction 했던 아이템들
    user_interactions = train_df.groupby(user_col)[item_col].agg(lambda x: set(x.tolist()))

    for uid, iid in zip(test_df[user_col].tolist(), test_df[item_col].tolist()):
        row = [uid, iid]

        try:
            inter_items = user_interactions[uid]
        except KeyError as e:
            inter_items = set([])
        n_items = len(inter_items)
        samples = random.choices(
            item_list, cum_weights=item_cumulate_count, k=n_items + n_sample + 100
        )

        sample = list(set(samples) - inter_items)
        sample = sample[:n_sample]
        assert len(sample) == n_sample
        row.extend(sample)

        negative_sampled_test.append(row)

    return negative_sampled_test


def loading_data(data_type: str) -> tuple[Any, list[list], Any, Any]:
    user_col = 'UserID'

    if data_type == '10M':
        item_col = 'MovieID'
        file_path = os.path.join(CONFIG.DATA, 'movielens', 'ml-10M100K')
        loading_function = loading_movielens_10m
    elif data_type == '1M':
        item_col = 'MovieID'
        file_path = os.path.join(CONFIG.DATA, 'movielens', 'ml-1m')
        loading_function = loading_movielens_1m
    elif data_type == 'BRUNCH':
        item_col = 'article_id'
        file_path = os.path.join(CONFIG.DATA, 'brunch_view')
        loading_function = loading_brunch
    else:
        raise ValueError(f"unknown data type {data_type}")

    train, test, item, user = loading_function(file_path)

    test_negative = get_negative_samples(train, test, user_col, item_col, n_sample=99)

    return train, test_negative, item, user


def movielens_preprocess(train: DataFrame, test: list, items: DataFrame, users: DataFrame) -> Tuple[
    DataFrame, DataFrame, DataFrame]:
    # UserID indexing
    user_id_mapper = DefaultDict(None, {
        user_id: user_index_id for user_index_id, user_id in enumerate(train['UserID'].unique())
    })

    train['user_id'] = train['UserID'].map(lambda x: user_id_mapper[x])
    users['user_id'] = users['UserID'].map(lambda x: user_id_mapper[x])

    # MovieID -> item_id
    item_id_mapper = DefaultDict(None, {
        movie_id: item_id for item_id, movie_id in enumerate(train['MovieID'].unique())
    })

    train['item_id'] = train['MovieID'].map(lambda x: item_id_mapper[x])
    items['item_id'] = items['MovieID'].map(lambda x: item_id_mapper[x])

    for i in range(len(test)):
        test[i][0] = user_id_mapper[test[i][0]]  # user id
        test[i][1:] = [item_id_mapper[item] for item in test[i][1:]]  # item id

    return train, test, items, users


def brunch_preprocess(train: DataFrame, test: list, items: DataFrame, users: DataFrame) -> Tuple[
    DataFrame, DataFrame, DataFrame]:
    return None


def preprocess_data(data_type: str, train: DataFrame, test: list, items: DataFrame, users: DataFrame) -> tuple[
    DataFrame, DataFrame, DataFrame]:

    if data_type == '10M':
        loading_function = movielens_preprocess
    elif data_type == '1M':
        loading_function = movielens_preprocess
    elif data_type == 'BRUNCH':
        loading_function = brunch_preprocess
    else:
        raise ValueError(f"unknown data type {data_type}")

    return loading_function(train, test, items, users)


if __name__ == '__main__':
    random.seed(42)
    argument = args()

    train_data, test_data, item_meta, user_meta = loading_data(argument.dataset)
    train_data, test_data, item_meta, user_meta = preprocess_data(
        argument.dataset, train_data, test_data, item_meta, user_meta
    )

    save_dir = os.path.join(CONFIG.DATA, argument.dataset)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    train_data.to_csv(os.path.join(save_dir, 'train.tsv'), sep='\t', index=False)
    item_meta.to_csv(os.path.join(save_dir, 'item_meta.tsv'), sep='\t', index=False)
    user_meta.to_csv(os.path.join(save_dir, 'user_meta.tsv'), sep='\t', index=False)

    with open(os.path.join(save_dir, 'negative_test.dat'), 'w') as f:
        for row in test_data:
            row = '\t'.join([str(v) for v in row])
            f.write(row + '\n')
