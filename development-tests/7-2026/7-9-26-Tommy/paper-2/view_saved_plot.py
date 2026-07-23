import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import pickle

PLOT_INDEX = 4
FIGURE_NAMES = ['semicircle_plot', 'rectified_semicircle_plot', 'linear_plot', 'disturbed_linear_plot', 'unrestricted_unit_representation']

fig = pickle.load(open(f'{FIGURE_NAMES[PLOT_INDEX]}.fig.pickle', 'rb'))
plt.show()
