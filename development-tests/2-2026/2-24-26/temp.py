import matplotlib.pyplot as plt
import numpy as np

no_pruning = np.array([
    [0, 94],
    [2, 94],
    [4, 94],
    [6, 98],
    [8, 96],
    [10, 94],
    [12, 88],
    [14, 90],
    [16, 90],
    [18, 90],
    [20, 72],
])

with_pruning = np.array([
    [0, 94],
    [2, 94],
    [4, 94],
    [6, 98],
    [8, 98],
    [10, 94],
    [12, 88],
    [14, 90],
    [16, 90],
    [18, 88],
    [20, 74],
])

plt.plot(no_pruning[:, 0], no_pruning[:, 1], color='#0088FF', label='Without Rule-Post Pruning')
plt.plot(with_pruning[:, 0], with_pruning[:, 1]+.15, color='#FF8800', label='With Rule-Post Pruning')
plt.xlim([-1, 21])
plt.xlabel('Corruption Percentage')
plt.ylim([70, 101])
plt.ylabel('Test Accuracy')
plt.legend()
plt.title("Corruption Percentage vs. Test Accuracy")
plt.grid(True)
plt.savefig('pruning-comparison.png', dpi=600)
plt.show()