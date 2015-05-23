[![Latest Version](https://pypip.in/version/pandas_confusion/badge.svg)](https://pypi.python.org/pypi/pandas_confusion/)
[![Supported Python versions](https://pypip.in/py_versions/pandas_confusion/badge.svg)](https://pypi.python.org/pypi/pandas_confusion/)
[![Download format](https://pypip.in/format/pandas_confusion/badge.svg)](https://pypi.python.org/pypi/pandas_confusion/)
[![License](https://pypip.in/license/pandas_confusion/badge.svg)](https://pypi.python.org/pypi/pandas_confusion/)
[![Development Status](https://pypip.in/status/pandas_confusion/badge.svg)](https://pypi.python.org/pypi/pandas_confusion/)
[![Downloads](https://pypip.in/download/pandas_confusion/badge.svg)](https://pypi.python.org/pypi/pandas_confusion/)
[![Code Health](https://landscape.io/github/scls19fr/pandas_confusion/master/landscape.svg?style=flat)](https://landscape.io/github/scls19fr/pandas_confusion/master)
[![Build Status](https://travis-ci.org/scls19fr/pandas_confusion.svg)](https://travis-ci.org/scls19fr/pandas_confusion)

# pandas_confusion
 
A [Python](https://www.python.org/) [Pandas](http://pandas.pydata.org/) implementation of [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).

WORK IN PROGRESS - Use it a your own risk

## Usage

## Confusion matrix

Let's define a (non binary) confusion matrix

    y_actu = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
    y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']

    confusion_matrix = ConfusionMatrix(y_actu, y_pred)
    print("Confusion matrix:\n%s" % confusion_matrix)

You can see it

    Predicted  cat  dog  rabbit
    Actual
    cat          3    0       0
    dog          0    1       2
    rabbit       2    1       3


### Matplotlib plot of a confusion matrix

    confusion_matrix.plot()
    plt.show()

![confusion_matrix](screenshots/cm.png)

### Matplotlib plot of a normalized confusion matrix

    confusion_matrix.plot(normalized=True)
    plt.show()

![confusion_matrix_norm](screenshots/cm_norm.png)

### Binary confusion matrix

    from pandas_confusion import BinaryConfusionMatrix, Backend

    y_actu = [ True,  True, False, False, False,  True, False,  True,  True,
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

    binary_confusion_matrix = BinaryConfusionMatrix(y_actu, y_pred)
    print("Binary confusion matrix:\n%s" % binary_confusion_matrix)


It display as a nicely labeled Pandas DataFrame

    Binary confusion matrix:
    Predicted  False  True
    Actual
    False         67      0
    True          21     24

You can get useful attributes such as True Positive (TP), True Negative (TN) ...

    print binary_confusion_matrix.TP

### Matplotlib plot of a binary confusion matrix

    binary_confusion_matrix.plot()
    plt.show()

![binary_confusion_matrix](screenshots/binary_cm.png)

### Matplotlib plot of a normalized binary confusion matrix

    binary_confusion_matrix.plot(normalized=True)
    plt.show()

![binary_confusion_matrix_norm](screenshots/binary_cm_norm.png)

### Seaborn plot of a binary confusion matrix (ToDo)

    binary_confusion_matrix.plot(backend=Backend.Seaborn)

### Confusion matrix and class statistics

Overall statistics and class statistics of confusion matrix can be easily displayed.

    y_true = [600, 200, 200, 200, 200, 200, 200, 200, 500, 500, 500, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    y_pred = [100, 200, 200, 100, 100, 200, 200, 200, 100, 200, 500, 100, 100, 100, 100, 100, 100, 100, 500, 200]
    cm = ConfusionMatrix(y_true, y_pred)
    cm.print_stats()

You should get:

    Confusion Matrix:

    Classes  100  200  500  600  __all__
    Actual
    100        0    0    0    0        0
    200        9    6    1    0       16
    500        1    1    1    0        3
    600        1    0    0    0        1
    __all__   11    7    2    0       20


    Overall Statistics:

    Accuracy: 0.35
    95% CI: (0.1539092047845412, 0.59218853453282805)
    No Information Rate: ToDo
    P-Value [Acc > NIR]: 0.978585644357
    Kappa: 0.0780141843972
    Mcnemar's Test P-Value: ToDo


    Class Statistics:

    Classes                                 100         200         500   600
    Population                               20          20          20    20
    Condition positive                        0          16           3     1
    Condition negative                       20           4          17    19
    Test outcome positive                    11           7           2     0
    Test outcome negative                     9          13          18    20
    TP: True Positive                         0           6           1     0
    TN: True Negative                         9           3          16    19
    FP: False Positive                       11           1           1     0
    FN: False Negative                        0          10           2     1
    TPR: Sensivity                          NaN       0.375   0.3333333     0
    TNR=SPC: Specificity                   0.45        0.75   0.9411765     1
    PPV: Pos Pred Value = Precision           0   0.8571429         0.5   NaN
    NPV: Neg Pred Value                       1   0.2307692   0.8888889  0.95
    FPR: False-out                         0.55        0.25  0.05882353     0
    FDR: False Discovery Rate                 1   0.1428571         0.5   NaN
    FNR: Miss Rate                          NaN       0.625   0.6666667     1
    ACC: Accuracy                          0.45        0.45        0.85  0.95
    F1 score                                  0   0.5217391         0.4     0
    MCC: Matthews correlation coefficient   NaN   0.1048285    0.326732   NaN
    Informedness                            NaN       0.125   0.2745098     0
    Markedness                                0  0.08791209   0.3888889   NaN
    Prevalence                                0         0.8        0.15  0.05
    LR+: Positive likelihood ratio          NaN         1.5    5.666667   NaN
    LR-: Negative likelihood ratio          NaN   0.8333333   0.7083333     1
    DOR: Diagnostic odds ratio              NaN         1.8           8   NaN
    FOR: False omission rate                  0   0.7692308   0.1111111  0.05


## ToDo list

* Matplotlib discrete colorbar

see ColorbarBase

http://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar

* Display numbers inside cells like http://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python

* Compare with results from Sklearn

    from sklearn.metrics import f1_score, classification_report
    f1_score(y_actu, y_pred)
    print classification_report(y_actu, y_pred)

* Compare with R "caret" package

http://stackoverflow.com/questions/26631814/create-a-confusion-matrix-from-a-dataframe

R

    Actual <- c(600, 200, 200, 200, 200, 200, 200, 200, 500, 500, 500, 200, 200, 200, 200, 200, 200, 200, 200, 200)
    Predicted <- c(100, 200, 200, 100, 100, 200, 200, 200, 100, 200, 500, 100, 100, 100, 100, 100, 100, 100, 500, 200)
    df <- data.frame(Actual, Predicted)
    #table(df)
    col <- sort(union(df$Actual, df$Predicted))
    df_conf <- table(lapply(df, factor, levels=col))
    #table(lapply(df, factor, levels=seq(100, 600, 100)))
    #table(lapply(df, factor, levels=c(100, 200, 500, 600)))

Python

    >>> from pandas_confusion import ConfusionMatrix
    >>> y_true = [600, 200, 200, 200, 200, 200, 200, 200, 500, 500, 500, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    >>> y_pred = [100, 200, 200, 100, 100, 200, 200, 200, 100, 200, 500, 100, 100, 100, 100, 100, 100, 100, 500, 200]
    >>> cm = ConfusionMatrix(y_true, y_pred)
    >>> cm
    Predicted  100  200  500  600  __all__
    Actual
    100          0    0    0    0        0
    200          9    6    1    0       16
    500          1    1    1    0        3
    600          1    0    0    0        1
    __all__     11    7    2    0       20

`cm(i, j)` in Python is `conf_mat(j, i)` in R

You can use `cm.to_dataframe().transpose()`

  * Overall statistics: No Information Rate, Mcnemar's Test P-Value

    see confusionMatrix.R and print.confusionMatrix.R (caret) and e1071 package

  * Class statistics

    * see Caret code for Detection Rate, Detection Prevalence, Balanced Accuracy


* Code metrics (landscape.io)

* Create fake truth, prediction from confusion matrix
(can be useful for unit test)

https://www.researchgate.net/post/Can_someone_help_me_to_calculate_accuracy_sensitivity_of_a_66_confusion_matrix

[see code (ToDo)](samples/fake_convol_mat.py)

* Order confusion matrix easily

* Create empty class easily

    cm = ConfusionMatrix(y_true, y_pred, labels=range(100, 600+1, 100))

    Class 300 and class 400 should be create

    R like method ? conf_mat_tab <- table(lapply(df, factor, levels = seq(100, 600, 100)))

    http://pandas.pydata.org/pandas-docs/stable/comparison_with_r.html

    idx_new_cls = pd.Index([300, 400])
    new_idx = df.index | idx_new_cls
    new_idx.name = 'Actual'
    new_col = df.index | idx_new_cls
    new_col.name = 'Predicted'
    df = df.loc[new_idx, new_col].fillna(0)

    see cm.enlarge(...)

* Calculate Mcnemar's Test P-Value with binary confusion matrix

    Actual <- c(TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE, TRUE,
            FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE,
            TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, TRUE, FALSE,
            TRUE, FALSE, FALSE, FALSE, FALSE, TRUE, TRUE, FALSE, FALSE,
            FALSE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE,
            TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
            FALSE, TRUE, TRUE, FALSE, TRUE, FALSE, TRUE, TRUE, TRUE,
            FALSE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE, FALSE,
            FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE,
            TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE, FALSE, TRUE,
            TRUE, FALSE, TRUE, FALSE, TRUE, FALSE, FALSE, TRUE, TRUE,
            FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE,
            FALSE, TRUE, TRUE, FALSE)

    Predicted <- c(FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE,
          FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
          TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE,
          FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE,
          FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
          TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
          FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE,
          FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE,
          FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE,
          FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, TRUE,
          TRUE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, TRUE,
          FALSE, FALSE, TRUE, TRUE, FALSE, FALSE, FALSE, FALSE, FALSE,
          FALSE, TRUE, FALSE, FALSE)

## Install

$ conda install pandas scikit-learn scipy

$ pip install pandas_confusion

## Done

* Continuous integration (Travis)

* Convert a confusion matrix to a binary confusion matrix

* Python package

* Unit tests (nose)

* Fix missing column and missing row

* Overall statistics: Accuracy, 95% CI, P-Value [Acc > NIR], Kappa