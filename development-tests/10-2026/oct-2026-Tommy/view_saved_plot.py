import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import pickle

PLOT_INDEX = 11
FIGURE_NAMES = {
    0 : 'semicircle_plot',
    1 : 'rectified_semicircle_plot',
    2 : 'linear_plot',
    3 : 'disturbed_linear_plot',
    4 : 'unrestricted_unit_representation',
    5 : 'unit_representation_pairwise_freezing',
    6 : 'unit_representation_anchor_freezing',
    7 : 'unit_representation_pairwise_joint',
    8 : 'unit_representation_anchor_joint',
    9 : 'unit_representation_pairwise_tune',
    10 : 'unit_representation_anchor_tune',
    11 : 'unit_representation_equidistant',
}

fig = pickle.load(open(f'{FIGURE_NAMES[PLOT_INDEX]}.fig.pickle', 'rb'))
plt.show()
