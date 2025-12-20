import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

MODE = 'decoupled'
BALANCE = 'high'

true = np.concatenate(([0]*1086,[0]*111,[1]*5,[1]*5))
predictions  = np.concatenate(([0]*1086,[1]*111,[0]*5,[1]*5))
cm = confusion_matrix(true, predictions)
cm_swapped = cm.T
disp = ConfusionMatrixDisplay(confusion_matrix=cm_swapped, display_labels=["Airplane", "Dog"])

fig, ax = plt.subplots()
disp.plot(ax=ax)

# Fix axis titles
ax.set_xlabel("True label")
ax.set_ylabel("Predicted label")
plt.savefig(f'confusion-matrix-{MODE}-{BALANCE}.png')
plt.show()