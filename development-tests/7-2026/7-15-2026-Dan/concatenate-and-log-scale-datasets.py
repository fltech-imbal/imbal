import os
import glob

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------

# The script assumes this layout:
#
# root/
# ├── data/
# │   ├── training/
# │   │   └── multiple CSV files
# │   ├── validation/
# │   │   └── multiple CSV files
# │   └── test/
# │       └── multiple CSV files
# └── scripts/
#     └── this_script.py

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
ROOT_DIRECTORY = os.path.dirname(SCRIPT_DIRECTORY)
DATA_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "fold0",)

# 10e-9 is equal to 1e-8.
EPSILON = 10e-9

# Columns listed here are completely removed before log scaling and will not
# appear in the generated CSV files.
COLUMNS_TO_EXCLUDE = [
    "Timestamp",
    "Event ID",
    "Target Timestamp",
    "Sunspot Number",
    "cme_donki_time",
    "CME_DONKI_latitude",
    "CME_DONKI_longitude",
    "CME_DONKI_speed",
    "CME_CDAW_MPA",
    "CME_CDAW_LinearSpeed",
    "VlogV",
    "DONKI_half_width",
    "Accelaration",
    "2nd_order_speed_final",
    "2nd_order_speed_20R",
    "CPA",
    "Halo",
    "Type2_Viz_Area",
    "solar_wind_speed",
    "diffusive_shock",
    "half_richardson_value",
    "CMEs Past Month",
    "CMEs Past 9 Hours",
    "CMEs Speed > 1000",
    "Max CME Speed",
    "delta_log_Intensity"
]

# Each enabled split is concatenated and written as one CSV.
SPLITS_TO_PROCESS = [
    "subtraining",
    "validation",
    "testing",
]

# Generated files are placed here.
OUTPUT_DIRECTORY = os.path.join(DATA_DIRECTORY, "log_scaled")

# Output filenames.
OUTPUT_FILENAMES = {
    "subtraining": "training_log_scaled.csv",
    "validation": "validation_log_scaled.csv",
    "testing": "test_log_scaled.csv",
}


def load_and_concatenate_csvs(input_directory):
    """
    Load every CSV file directly inside input_directory and concatenate them
    row-wise in sorted filename order.
    """
    csv_paths = sorted(
        glob.glob(os.path.join(input_directory, "*.csv"))
    )

    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files were found in: {input_directory}"
        )

    dataframes = []

    for csv_path in csv_paths:
        dataframe = pd.read_csv(csv_path)

        print(
            f"Loaded {os.path.basename(csv_path)}: "
            f"{len(dataframe)} rows"
        )

        dataframes.append(dataframe)

    combined_dataframe = pd.concat(
        dataframes,
        axis=0,
        ignore_index=True,
    )

    print(
        f"Concatenated {len(csv_paths)} files into "
        f"{len(combined_dataframe)} total rows."
    )

    return combined_dataframe


def remove_excluded_columns(dataframe, columns_to_exclude):
    """
    Remove configured columns. Missing configured columns produce a warning
    rather than stopping execution.
    """
    existing_columns = [
        column
        for column in columns_to_exclude
        if column in dataframe.columns
    ]

    missing_columns = [
        column
        for column in columns_to_exclude
        if column not in dataframe.columns
    ]

    if missing_columns:
        print(
            "Warning: The following excluded columns were not found: "
            f"{missing_columns}"
        )

    if existing_columns:
        print(f"Removing columns: {existing_columns}")
        dataframe = dataframe.drop(columns=existing_columns)

    return dataframe


def validate_dataframe_for_log_scaling(dataframe, split_name):
    """
    Confirm that every remaining column is numeric, finite, nonmissing, and
    greater than -EPSILON so that ln(value + EPSILON) is defined.
    """
    if dataframe.empty:
        raise ValueError(
            f"The '{split_name}' dataset is empty after concatenation."
        )

    nonnumeric_columns = dataframe.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()

    if nonnumeric_columns:
        raise ValueError(
            f"The '{split_name}' dataset contains nonnumeric columns: "
            f"{nonnumeric_columns}. Exclude or encode these columns first."
        )

    columns_with_missing_values = dataframe.columns[
        dataframe.isna().any()
    ].tolist()

    if columns_with_missing_values:
        raise ValueError(
            f"The '{split_name}' dataset contains missing values in: "
            f"{columns_with_missing_values}"
        )

    values = dataframe.to_numpy(dtype=np.float64)

    if not np.isfinite(values).all():
        invalid_positions = np.argwhere(~np.isfinite(values))
        first_row, first_column = invalid_positions[0]

        raise ValueError(
            f"The '{split_name}' dataset contains a non-finite value at "
            f"row {first_row}, column "
            f"'{dataframe.columns[first_column]}'."
        )

    invalid_mask = values + EPSILON <= 0

    if invalid_mask.any():
        invalid_positions = np.argwhere(invalid_mask)
        first_row, first_column = invalid_positions[0]
        invalid_value = values[first_row, first_column]

        raise ValueError(
            f"Natural-log scaling is undefined for value {invalid_value} "
            f"in the '{split_name}' dataset at row {first_row}, column "
            f"'{dataframe.columns[first_column]}', because "
            f"value + EPSILON must be greater than zero."
        )


def apply_log_scaling(dataframe):
    """
    Apply ln(value + EPSILON) to every remaining value while preserving the
    original column names and row order.
    """
    scaled_values = np.log(
        dataframe.to_numpy(dtype=np.float64) + EPSILON
    )

    return pd.DataFrame(
        scaled_values,
        columns=dataframe.columns,
        index=dataframe.index,
    )


def process_split(split_name):
    """
    Concatenate, filter, log-scale, and save one dataset split.
    """
    input_directory = os.path.join(
        DATA_DIRECTORY,
        split_name,
    )

    if split_name not in OUTPUT_FILENAMES:
        raise ValueError(
            f"No output filename is configured for split '{split_name}'."
        )

    print("\n" + "=" * 72)
    print(f"Processing {split_name} dataset")
    print("=" * 72)

    dataframe = load_and_concatenate_csvs(
        input_directory
    )

    dataframe = remove_excluded_columns(
        dataframe,
        COLUMNS_TO_EXCLUDE,
    )

    validate_dataframe_for_log_scaling(
        dataframe,
        split_name,
    )

    log_scaled_dataframe = apply_log_scaling(
        dataframe
    )

    os.makedirs(
        OUTPUT_DIRECTORY,
        exist_ok=True,
    )

    output_path = os.path.join(
        OUTPUT_DIRECTORY,
        OUTPUT_FILENAMES[split_name],
    )

    log_scaled_dataframe.to_csv(
        output_path,
        index=False,
    )

    print(f"Columns written: {len(log_scaled_dataframe.columns)}")
    print(f"Rows written: {len(log_scaled_dataframe)}")
    print(f"Saved log-scaled dataset to: {output_path}")

    return output_path


def main():
    generated_files = []

    for split_name in SPLITS_TO_PROCESS:
        generated_files.append(
            process_split(split_name)
        )

    print("\n" + "=" * 72)
    print("Finished generating log-scaled datasets")
    print("=" * 72)

    for generated_file in generated_files:
        print(generated_file)


if __name__ == "__main__":
    main()
