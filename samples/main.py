#!/usr/bin/python
# -*- coding: utf8 -*-

import pandas as pd
from pandas_confusion import BinaryConfusionMatrix

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

confusion_matrix = BinaryConfusionMatrix(y_actu, y_pred)
print("Confusion matrix:\n%s" % confusion_matrix)


attributes = ['TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'PPV', 'NPV', 'FPR', 'FDR',
    'FNR', 'ACC', 'F1_score', 'MCC', 'informedness', 'markedness']
for attrib in attributes:
    print("%s: %f" % (attrib, getattr(confusion_matrix, attrib)))

#confusion_matrix.plot()
