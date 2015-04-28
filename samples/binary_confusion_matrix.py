#!/usr/bin/python
# -*- coding: utf8 -*-

import os
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pandas_confusion import BinaryConfusionMatrix, Backend

def main():
    basepath = os.path.dirname(__file__)

    y_true = [ True,  True, False, False, False,  True, False,  True,  True,
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
           False,  True,  True, False]
    
    y_pred = [False, False, False, False, False,  True, False, False,  True,
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
           False,  True, False, False]

    binary_confusion_matrix = BinaryConfusionMatrix(y_true, y_pred)
    print("Binary confusion matrix:\n%s" % binary_confusion_matrix)

    print("")

    attributes = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'PPV', 'NPV', 'FPR', 'FDR',
        'FNR', 'ACC', 'F1_score', 'MCC', 'informedness', 'markedness']
    for attrib in attributes:
        print("%s: %f" % (attrib, getattr(binary_confusion_matrix, attrib)))

    binary_confusion_matrix.plot()
    filename = 'binary_confusion_matrix.png'
    plt.savefig(os.path.join(basepath, '..','screenshots', filename))
    plt.show()

    binary_confusion_matrix.plot(normalized=True)
    filename = 'binary_confusion_matrix_norm.png'
    plt.savefig(os.path.join(basepath, '..','screenshots', filename))
    plt.show()

    #binary_confusion_matrix.plot(normalized=True, backend=Backend.Seaborn)
    #plt.show()

if __name__ == "__main__":
    main()
