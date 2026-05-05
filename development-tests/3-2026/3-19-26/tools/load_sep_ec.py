import pandas as pd
import numpy as np
import glob
import os

def load_sep_ec(
    path,
    include_cme=True,
    normalize_data=True,
    return_normalization_factors=False,
    normalization_factors=None
):
    """
    Load data
    """

    files = [f for f in glob.glob(os.path.join(path, "*")) if os.path.isfile(f)]

    def load_multiple_csv(csv_files):
        loaded_data = None
        for file in csv_files:
            df = pd.read_csv(file)
            if loaded_data is None:
                loaded_data = df
            else:
                loaded_data = pd.concat([loaded_data, df])
        return loaded_data

    data = load_multiple_csv(files)

    if data is None:
        raise Exception("Unable to load data (specified directory is likely incorrect).")

    labels = data.pop("delta_log_Intensity")

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

    data = clean_sep_ec_data(data)

    data = data.to_numpy()
    labels = labels.to_numpy()

    normalization_factors = None
    if normalize_data:
        if normalization_factors is None:
            absolute_max_values = np.abs(data).max(axis=0)
        else:
            absolute_max_values = normalization_factors
        if return_normalization_factors:
            normalization_factors = absolute_max_values
        data = np.divide(data, absolute_max_values)

    if return_normalization_factors:
        return data, labels, normalization_factors
    else:
        return data, labels

