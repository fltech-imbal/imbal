from matplotlib import pyplot as plt
import os

modes = ['regular', 'balanced', 'decoupled']
options = ['_', '_w_validation', '_w_validation_ae', '_w_validation_ae_third_last']

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(53.33, 40))

prefix = 'sep_ec_log_normalized'
OUTPUT_PATH = 'dtw-results'

for i, mode in enumerate(modes):
    for j, option in enumerate(options):
        if i == 0:
            ax[i, j].set_title(['Basic', 'w/ Validation', 'w/ Validation, AE', 'w/ Validation, AE-3'][j], fontsize=50)
        path = f'{OUTPUT_PATH}/{prefix}_{mode}{option}.png'
        print(path)
        if os.path.isfile(path):
            img = plt.imread(path)
            ax[i, j].imshow(img)
            ax[i, j].axis('off')

for axx, row in zip(ax[:,0], ['Regular', 'Balanced', 'Decoupled']):
    axx.text(
        -0, 0.5, row,
        transform=axx.transAxes,
        ha='center',
        va='center',
        fontsize=50,
        rotation=90
    )

plt.tight_layout()
plt.savefig(f'{OUTPUT_PATH}/{prefix}_summary.png')
plt.show()