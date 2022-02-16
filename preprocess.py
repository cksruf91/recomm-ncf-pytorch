import argparse
import os
from typing import Tuple

from pandas import DataFrame

from common.loading_functions import loading_data
from config import CONFIG


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='10M', choices=['10M', '1M', 'BRUNCH'], help='데이터셋', type=str)
    return parser.parse_args()


def movielens_preprocess(train: DataFrame, test: list, items: DataFrame, users: DataFrame) -> Tuple[
    DataFrame, list, DataFrame, DataFrame]:
    train = train[['item_id', 'user_id', 'Rating']]
    items = items[["MovieID", "Title", "Genres", "item_id"]]
    return train, test, items, users


def brunch_preprocess(train: DataFrame, test: list, items: DataFrame, users: DataFrame) -> Tuple[
    DataFrame, list, DataFrame, DataFrame]:
    print(f"train dataset : {len(train)}")

    train['Rating'] = 5  # 사용자가 해당글을 좋아한다고 가정

    train = train[['item_id', 'user_id', 'Rating', 'Timestamp']]
    items = items[["item_id", "magazine_id", "user_id", "title", "sub_title", "keyword_list", "display_url", "id"]]
    users = users[['user_id', 'keyword_list', 'following_list', 'id']]
    return train, test, items, users


def preprocess_data(data_type: str, train: DataFrame, test: list, items: DataFrame, users: DataFrame) -> Tuple[
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
