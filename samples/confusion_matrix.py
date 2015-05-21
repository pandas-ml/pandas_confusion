#!/usr/bin/python
# -*- coding: utf8 -*-

import os
#import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pandas_confusion import ConfusionMatrix, Backend

def main():
    basepath = os.path.dirname(__file__)

    #y_true = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
    #y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
    ##cm = ConfusionMatrix(y_true, y_pred)
    #cm = ConfusionMatrix(y_true, y_pred, labels=["ant", "bird", "cat"])

    #y_true = [2, 0, 2, 2, 0, 1]
    #y_pred = [0, 0, 2, 2, 0, 2]
    ##cm = ConfusionMatrix(y_true, y_pred)
    #cm = ConfusionMatrix(y_true, y_pred, labels=["ant", "bird", "cat"])

    y_true = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
    y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']
    cm = ConfusionMatrix(y_true, y_pred)

    #y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    #y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    #>>> cm(y_true, y_pred, labels=["ant", "bird", "cat"])
    #array([[2, 0, 0],
    #       [0, 0, 1],
    #       [1, 0, 2]])
    #cm = ConfusionMatrix(y_true, y_pred)

    print("Confusion matrix:\n%s" % cm)

    cm.plot()
    #filename = 'cm.png'
    #plt.savefig(os.path.join(basepath, '..','screenshots', filename))
    #plt.show()

    cm.plot(normalized=True)
    #filename = 'cm_norm.png'
    #plt.savefig(os.path.join(basepath, '..','screenshots', filename))
    #plt.show()

    #import seaborn as sns
    #cm.plot(normalized=True, backend=Backend.Seaborn)
    #sns.plt.show()

    # Binarize a confusion matrix
    y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    cm = ConfusionMatrix(y_true, y_pred)
    print(cm)
    binary_cm = cm.binarize(['ant', 'cat'])
    # A bird is not a "land_animal"
    print(binary_cm)

    df = pd.DataFrame({
    'Height': [150, 150, 151, 151, 152, 155, 155, 157, 157, 157, 157, 158, 158, 159, 159, 159, 160, 160, 162, 162, 163, 164, 165, 168, 169, 169, 169, 170, 171, 171, 173, 173, 174, 176, 177, 177, 179, 179, 179, 179, 179, 181, 181, 182, 183, 184, 186, 190, 190],
    'Weight': [54, 55, 55, 47, 58, 53, 59, 60, 56, 55, 62, 56, 55, 55, 64, 61, 59, 59, 63, 66, 64, 62, 66, 66, 72, 65, 75, 71, 70, 70, 75, 65, 79, 78, 83, 75, 84, 78, 74, 75, 74, 90, 80, 81, 90, 81, 91, 87, 100],
    'Size': ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL'],
    'SizePred': ['L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'S', 'XL', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'M', 'S', 'S', 'M', 'S', 'M', 'M', 'M']
    })
    cm = ConfusionMatrix(df["Size"], df["SizePred"])
    print(cm)
    assert (cm.true - df.Size.value_counts()).sum() == 0
    assert (cm.pred - df.SizePred.value_counts()).sum() == 0
    #df.SizePred.value_counts()

if __name__ == "__main__":
    main()