import argparse
import os
import random
from itertools import accumulate
from typing import Any

from pandas import DataFrame

from common.loading_functions import loading_brunch, loading_movielens_1m, loading_movielens_10m
from config import CONFIG


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='10M', choices=['10M', '1M', 'BRUNCH'], help='데이터셋', type=str)
    return parser.parse_args()


def uniform_random_sample(n, exclude_items, items):
    sample = []
    while len(sample) < n:
        n_item = random.choice(items)
        if n_item in exclude_items:
            continue
        if n_item in sample:
            continue
        sample.append(n_item)
    assert len(sample) == n
    return sample


def weighted_random_sample(n, exclude_items, items, cum_sums):
    n_items = len(exclude_items)
    samples = random.choices(
        items, cum_weights=cum_sums, k=n_items + n + 100
    )

    sample = list(set(samples) - exclude_items)
    sample = sample[:n]
    assert len(sample) == n
    return sample


def get_negative_samples(train_df, test_df, user_col, item_col, n_sample=99, method='random'):
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

        if method == 'random':
            sample = uniform_random_sample(n_sample, inter_items, item_list)
        elif method == 'weighted':
            sample = weighted_random_sample(n_sample, inter_items, item_list, cum_sums=item_cumulate_count)
        else:
            raise ValueError(f"invalid sampling method {method}")

        row.extend(sample)

        negative_sampled_test.append(row)

    return negative_sampled_test


def loading_data(data_type: str) -> tuple[Any, list[list], Any, Any]:
    user_col = 'user_id'

    if data_type == '10M':
        item_col = 'item_id'
        file_path = os.path.join(CONFIG.DATA, 'movielens', 'ml-10M100K')
        loading_function = loading_movielens_10m
    elif data_type == '1M':
        item_col = 'item_id'
        file_path = os.path.join(CONFIG.DATA, 'movielens', 'ml-1m')
        loading_function = loading_movielens_1m
    elif data_type == 'BRUNCH':
        item_col = 'article_id'
        file_path = os.path.join(CONFIG.DATA, 'brunch_view')
        loading_function = loading_brunch
    else:
        raise ValueError(f"unknown data type {data_type}")

    train, test, item, user = loading_function(file_path)

    test_negative = get_negative_samples(train, test, user_col, item_col, n_sample=99, method='random')

    return train, test_negative, item, user


def movielens_preprocess(train: DataFrame, test: list, items: DataFrame, users: DataFrame) -> tuple[
    DataFrame, list, DataFrame, DataFrame]:
    train = train[['item_id', 'user_id', 'Rating']]
    items = items[["MovieID", "Title", "Genres", "item_id"]]
    users = users[["UserID", "user_id", "Gender", "Age", "Occupation", "Zip-code"]]
    return train, test, items, users


def brunch_preprocess(train: DataFrame, test: list, items: DataFrame, users: DataFrame) -> tuple[
    DataFrame, list, DataFrame, DataFrame]:
    return train, test, items, users


def preprocess_data(data_type: str, train: DataFrame, test: list, items: DataFrame, users: DataFrame) -> tuple[
    DataFrame, list, DataFrame, DataFrame]:
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
