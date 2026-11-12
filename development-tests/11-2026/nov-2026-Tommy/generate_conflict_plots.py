import os
import tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODE = 'C'

if MODE == 'E':
    PATH = 'results-sep-e'
    OUTPUT_PATH = 'results-sep-e/cosine-similarities'
    DATA_PATH = 'cleaned-dtw-SEP-EC-data'
    DATA_PREFIX = 'sep_e_log_normalized'
    data_load_function = tools.load_sep_ec_data
    BATCHES_PER_EPOCH = 8
else:
    PATH = 'results-sep-c'
    OUTPUT_PATH = 'results-sep-c/cosine-similarities'
    DATA_PATH = 'cleaned-SEP-C-data'
    DATA_PREFIX = 'sep_c'
    data_load_function = tools.load_sep_c_data
    BATCHES_PER_EPOCH = 3


# (x_train, y_train), (x_val, y_val), (x_test, y_test) = data_load_function(
#     f"{DATA_PATH}/{DATA_PREFIX}",
# )

# y_test = y_test.reshape(-1)
# print(np.min(y_test), np.max(y_test))

count = 0
for index, filename in enumerate(os.listdir(PATH)):
    if not '.' in filename:
        continue

    if 'clustered' not in filename:
        continue
    count += 1
    # if count <= 15:
    #     continue

    file_prefix = filename.split('.')[0]

    bin_similarities = [[] for i in range(3)]
    epochs = []
    with open(f"{PATH}/{filename}", 'r') as f:
        for line in f.readlines():
            values = line.split(',')
            try:
                values = [float(x) for x in values]
            except ValueError:
                continue
            if values[1] != BATCHES_PER_EPOCH:
                continue
            bin_similarities[0].append(values[8])
            bin_similarities[1].append(values[9])
            bin_similarities[2].append(values[10])
            epochs.append(values[0])

    for i in range(3):
        bin_similarities[i] = pd.Series(bin_similarities[i]).rolling(window=10, center=True, min_periods=1).mean()


    bin_similarities = np.array(bin_similarities)
    plt.plot(epochs, bin_similarities[0], label=f'Bin 0')
    plt.plot(epochs, bin_similarities[1], label=f'Bin 1')
    plt.plot(epochs, bin_similarities[2], label=f'Bin 2')
    plt.ylabel('Cosine Similarity')
    plt.xlabel('Epoch')
    plt.legend()
    plt.ylim(bottom=min(0.0, np.min(bin_similarities)))
    plt.savefig(f"{OUTPUT_PATH}/{file_prefix}-cosine-similarities.png")
    plt.show()




