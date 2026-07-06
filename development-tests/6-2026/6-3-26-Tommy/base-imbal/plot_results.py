from matplotlib import pyplot as plt

modes = ['regular', 'balanced', 'decoupled']
options = ['_', '_w_validation', '_w_validation_ae', '_w_validation_ae_third_last']

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(53.33, 40))

prefix = 'sep_e_log_normalized'

for i, mode in enumerate(modes):
    for j, option in enumerate(options):
        if i == 0:
            ax[i, j].set_title(['Basic', 'w/ Validation', 'w/ Validation, AE', 'w/ Validation, AE-3'][j], fontsize=50)
        img = plt.imread(f'results/{prefix}_{mode}{option}.png')
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
plt.savefig(f'results/{prefix}_summary.png')
plt.show()