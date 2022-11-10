import pandas as pd
import numpy as np
import logging

TRAIN_CSV_PATH = 'facial-keypoints-detection/training/training.csv'
TEST_CSV_PATH = 'facial-keypoints-detection/test/test.csv'


def load_csv(path):
    _csv = pd.read_csv(path)
    logging.info('{} has length - {}'.format(path, len(_csv)))
    logging.info('{}'.format(_csv.info()))
    return _csv


def clean_csv(i_csv: pd.DataFrame):
    _csv_allValid = i_csv.dropna()
    _csv_autoFill = i_csv.ffill()
    _csv_missingOnly = i_csv[i_csv.isna().any(axis=1)]

    logging.info('All Valid Shape - {}\nAuto Fill Shape - {}\nMissing Only Shape - {}'.format(
        _csv_allValid.shape, _csv_autoFill.shape, _csv_missingOnly.shape
    ))

    return _csv_allValid, _csv_autoFill, _csv_missingOnly


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    train_csv = load_csv(TRAIN_CSV_PATH)
    csv_allValid, csv_autoFill, csv_missingOnly = clean_csv(train_csv)
