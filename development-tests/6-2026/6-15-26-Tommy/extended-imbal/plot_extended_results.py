from matplotlib import pyplot as plt
import os

modes = ['balanced', 'decoupled', 'balanced', 'decoupled']
options = ['_w_validation', '_w_validation_ae', '_w_validation_ae_third_last']

fig, ax = plt.subplots(nrows=4, ncols=3, figsize=(40, 53.33))

prefix = 'sep_e_no_electron_log_normalized'

for i, mode in enumerate(modes):
    for j, option in enumerate(options):
        if i < 2:
            suffix = '_mdi'
        else:
            suffix = '_denseweight'
        if i == 0:
            ax[i, j].set_title(['w/ Validation', 'w/ Validation, AE', 'w/ Validation, AE-3'][j], fontsize=50)

        path = f'results/{prefix}_{mode}{option}{suffix}.png'
        if os.path.isfile(path):
            img = plt.imread(path)
            ax[i, j].imshow(img)
            ax[i, j].axis('off')

for axx, row in zip(ax[:,0], ['Balanced (MDI)', 'Decoupled (MDI)', 'Balanced (DW)', 'Decoupled (DW)']):
    axx.text(
        -0, 0.5, row,
        transform=axx.transAxes,
        ha='center',
        va='center',
        fontsize=50,
        rotation=90
    )

plt.tight_layout()
plt.savefig(f'results/{prefix}_extended_summary.png')
plt.show()