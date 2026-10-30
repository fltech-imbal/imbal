import pandas as pd
import numpy as np
import glob
import os

def load_sep_ec(
    path,
    include_cme=True,
    normalize_data=True,
    use_delta=True
):
    """
    Load data
    """

    training_files = [f for f in glob.glob(os.path.join(path, "training/*")) if os.path.isfile(f)]
    test_files = [f for f in glob.glob(os.path.join(path, "testing/*")) if os.path.isfile(f)]

    def load_multiple_csv(csv_files):
        loaded_data = None
        for file in csv_files:
            df = pd.read_csv(file)
            if loaded_data is None:
                loaded_data = df
            else:
                loaded_data = pd.concat([loaded_data, df])
        return loaded_data

    training_data = load_multiple_csv(training_files)
    test_data = load_multiple_csv(test_files)

    if (training_data is None) or (test_data is None):
        raise Exception("Unable to load data (specified directory is likely incorrect).")


    training_labels = training_data.pop("delta_log_Intensity")
    test_labels = test_data.pop("delta_log_Intensity")

    def clean_sep_ec_data(
        df
    ):
        df = df.select_dtypes(include=[np.number])
        df = df.dropna()
        df = df.drop(columns=["Event ID"])
        if not include_cme:
            cme_start_index = df.columns.get_loc("Sunspot Number")
            df = df.drop(df.columns[cme_start_index:], axis=1)

        return df

    training_data = clean_sep_ec_data(training_data)
    test_data = clean_sep_ec_data(test_data)

    if normalize_data:
        absolute_max_values = training_data.abs().max()
        test_data = test_data.div(absolute_max_values, axis=1)
        training_data = training_data.div(absolute_max_values, axis=1)

    training_data = training_data.to_numpy()
    training_labels = training_labels.to_numpy()
    test_data = test_data.to_numpy()
    test_labels = test_labels.to_numpy()

    return (training_data, training_labels), (test_data, test_labels)