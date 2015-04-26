#!/usr/bin/python
# -*- coding: utf8 -*-

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pandas_confusion import BinaryConfusionMatrix, Backend

def main():
    y_actu = pd.Series([ True,  True, False, False, False,  True, False,  True,  True,
           False,  True, False, False, False, False, False,  True, False,
            True,  True,  True,  True, False, False, False,  True, False,
            True, False, False, False, False,  True,  True, False, False,
           False,  True,  True,  True,  True, False, False, False, False,
            True, False, False, False, False, False, False, False, False,
           False,  True,  True, False,  True, False,  True,  True,  True,
           False, False,  True, False,  True, False, False,  True, False,
           False, False, False, False, False, False, False,  True, False,
            True,  True,  True,  True, False, False,  True, False,  True,
            True, False,  True, False,  True, False, False,  True,  True,
           False, False,  True,  True, False, False, False, False, False,
           False,  True,  True, False])
    y_pred = pd.Series([False, False, False, False, False,  True, False, False,  True,
           False,  True, False, False, False, False, False, False, False,
            True,  True,  True,  True, False, False, False, False, False,
           False, False, False, False, False,  True, False, False, False,
           False,  True, False, False, False, False, False, False, False,
            True, False, False, False, False, False, False, False, False,
           False,  True, False, False, False, False, False, False, False,
           False, False,  True, False, False, False, False,  True, False,
           False, False, False, False, False, False, False,  True, False,
           False,  True, False, False, False, False,  True, False,  True,
            True, False, False, False,  True, False, False,  True,  True,
           False, False,  True,  True, False, False, False, False, False,
           False,  True, False, False])

    binary_confusion_matrix = BinaryConfusionMatrix(y_actu, y_pred)
    print("Binary confusion matrix:\n%s" % binary_confusion_matrix)

    attributes = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'PPV', 'NPV', 'FPR', 'FDR',
        'FNR', 'ACC', 'F1_score', 'MCC', 'informedness', 'markedness']
    for attrib in attributes:
        print("%s: %f" % (attrib, getattr(binary_confusion_matrix, attrib)))

    binary_confusion_matrix.plot()
    plt.show()

    binary_confusion_matrix.plot(normalized=True)
    plt.show()

    binary_confusion_matrix.plot(normalized=True, backend=Backend.Seaborn)
    plt.show()

if __name__ == "__main__":
    main()
