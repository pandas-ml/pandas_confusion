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


if __name__ == "__main__":
    main()