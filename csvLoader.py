import pandas as pd
import numpy as np
import logging

TRAIN_CSV_PATH = 'facial-keypoints-detection/training/training.csv'
TEST_CSV_PATH = 'facial-keypoints-detection/test/test.csv'


def load_csv(path):
    """
    Load the csv file from path into pd dataframe

    Args:
        path:       Path to the local csv data file

    Returns:        pd.DataFrame    -   pd dataframe of all data

    """
    _csv = pd.read_csv(path)
    logging.info('{} has length - {}'.format(path, len(_csv)))
    logging.info('{}'.format(_csv.info()))
    return _csv


def getImgArrAndKptsDf(i_df):
    """
    Split input df into img data array and kpts data array.

    Args:
        i_df:       Input pd.Dataframe of all data to split

    Returns:        List    -   image data np array, kpts data dataframe.

    """
    # Get img data
    img_arr = np.array(i_df.Image)
    for i in range(len(img_arr)):
        img_arr[i] = np.fromstring(img_arr[i], sep=' ')

    # Get kpts data
    kpts_df = i_df.drop(['Image'], axis=1)
    return img_arr, kpts_df


def clean_csv(i_csv: pd.DataFrame):
    """
    Process the input dataframe into 3 different dfs, one with all valid data,
    second with auto-filled data and the last with only rows with missing data.

    Args:
        i_csv:      pd dataframe to process

    Returns:        List[dfs]    -  one with all valid data, second with auto-filled
                                    data and the last with only rows with missing data.

    """
    _csv_allValid = i_csv.dropna()
    _csv_autoFill = i_csv.ffill()
    _csv_missingOnly = i_csv[i_csv.isna().any(axis=1)]

    logging.info('All Valid Shape - {}\nAuto Fill Shape - {}\nMissing Only Shape - {}'.format(
        _csv_allValid.shape, _csv_autoFill.shape, _csv_missingOnly.shape
    ))

    return _csv_allValid, _csv_autoFill, _csv_missingOnly


if __name__ == '__main__':
    # logging.getLogger().setLevel(logging.INFO)
    # train_csv = load_csv(TRAIN_CSV_PATH)
    # print("Len of train csv: " + str(len(np.array(train_csv.Image))))
    # csv_allValid, csv_autoFill, csv_missingOnly = clean_csv(train_csv)
    mean = (0.5,)
    print(mean)
