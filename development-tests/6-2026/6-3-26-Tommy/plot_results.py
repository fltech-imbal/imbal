from matplotlib import pyplot as plt

modes = ['regular', 'balanced', 'decoupled']
options = ['_', '_w_validation', '_w_validation_ae']

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(40, 40))

prefix = 'sep_e_no_electron_log_normalized'

for i, mode in enumerate(modes):
    for j, option in enumerate(options):
        if i == 0:
            ax[i, j].set_title(['Basic', 'w/ Validation', 'w/ Validation, AE'][j], fontsize=50)
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