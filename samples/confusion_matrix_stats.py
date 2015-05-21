#!/usr/bin/python
# -*- coding: utf8 -*-

import os
#import seaborn as sns
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from pandas_confusion import ConfusionMatrix, Backend

def size_pred():
    df = pd.DataFrame({
        'Height': [150, 150, 151, 151, 152, 155, 155, 157, 157, 157, 157, 158, 158, 159, 159, 159, 160, 160, 162, 162, 163, 164, 165, 168, 169, 169, 169, 170, 171, 171, 173, 173, 174, 176, 177, 177, 179, 179, 179, 179, 179, 181, 181, 182, 183, 184, 186, 190, 190],
        'Weight': [54, 55, 55, 47, 58, 53, 59, 60, 56, 55, 62, 56, 55, 55, 64, 61, 59, 59, 63, 66, 64, 62, 66, 66, 72, 65, 75, 71, 70, 70, 75, 65, 79, 78, 83, 75, 84, 78, 74, 75, 74, 90, 80, 81, 90, 81, 91, 87, 100],
        'Size': ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL'],
        'SizePred': ['S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'S', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'XL', 'L', 'XL', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'L', 'M', 'L', 'L', 'M', 'L', 'M', 'M', 'M']
    })
    cm = ConfusionMatrix(df["Size"], df["SizePred"])
    print(cm)

    #cm.print_stats()
    #ToFix
    #TypeError: ('integer argument expected, got float', u'occurred at index L')

    cm.plot()
    filename = 'size.png'
    plt.savefig(os.path.join(basepath, '..','screenshots', filename))
    plt.show()

def number_pred():
    y_true = [600, 200, 200, 200, 200, 200, 200, 200, 500, 500, 500, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    y_pred = [100, 200, 200, 100, 100, 200, 200, 200, 100, 200, 500, 100, 100, 100, 100, 100, 100, 100, 500, 200]
    cm = ConfusionMatrix(y_true, y_pred)

    #print(cm.binarize(100).P)
    #cm.enlarge(300)
    #cm.enlarge([300, 400])

    #print(cm)

    #print("")

    #print(cm.classes)

    print("")

    #cm.print_stats(None)
    cm.print_stats()

def main():
    size_pred()
    #number_pred()

if __name__ == "__main__":
    basepath = os.path.dirname(__file__)
    main()