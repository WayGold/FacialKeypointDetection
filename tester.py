from dataLoader import *

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    train_csv = load_csv(TRAIN_CSV_PATH)
    print("Len of train csv: " + str(len(np.array(train_csv.Image))))
    csv_allValid, csv_autoFill, csv_missingOnly = clean_csv(train_csv)

    test_ds = FacialKptsDataSet(train_csv)
