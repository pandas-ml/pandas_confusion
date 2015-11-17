|Latest Version| |Supported Python versions| |Wheel format| |License|
|Development Status| |Downloads monthly| |Requirements Status| |Code
Health| |Codacy Badge| |Build Status|

pandas\_confusion
=================

A `Python <https://www.python.org/>`__
`Pandas <http://pandas.pydata.org/>`__ implementation of `confusion
matrix <https://en.wikipedia.org/wiki/Confusion_matrix>`__.

WORK IN PROGRESS - Use it a your own risk

Usage
-----

Confusion matrix
----------------

Import ``ConfusionMatrix``

::

    from pandas_confusion import ConfusionMatrix

Define actual values (``y_actu``) and predicted values (``y_pred``)

::

    y_actu = ['rabbit', 'cat', 'rabbit', 'rabbit', 'cat', 'dog', 'dog', 'rabbit', 'rabbit', 'cat', 'dog', 'rabbit']
    y_pred = ['cat', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'dog', 'cat', 'rabbit', 'cat', 'rabbit', 'rabbit']

Let's define a (non binary) confusion matrix

::

    confusion_matrix = ConfusionMatrix(y_actu, y_pred)
    print("Confusion matrix:\n%s" % confusion_matrix)

You can see it

::

    Predicted  cat  dog  rabbit  __all__
    Actual
    cat          3    0       0        3
    dog          0    1       2        3
    rabbit       2    1       3        6
    __all__      5    2       5       12

Matplotlib plot of a confusion matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Inside a IPython notebook add this line as first cell

::

    %matplotlib inline

You can plot confusion matrix using:

::

    import matplotlib.pyplot as plt

    confusion_matrix.plot()

If you are not using inline mode, you need to use to show confusion
matrix plot.

::

    plt.show()

.. figure:: screenshots/cm.png
   :alt: confusion\_matrix

   confusion\_matrix

Matplotlib plot of a normalized confusion matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    confusion_matrix.plot(normalized=True)
    plt.show()

.. figure:: screenshots/cm_norm.png
   :alt: confusion\_matrix\_norm

   confusion\_matrix\_norm

Binary confusion matrix
~~~~~~~~~~~~~~~~~~~~~~~

Import ``BinaryConfusionMatrix`` and ``Backend``

::

    from pandas_confusion import BinaryConfusionMatrix, Backend

Define actual values (``y_actu``) and predicted values (``y_pred``)

::

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

Let's define a binary confusion matrix

::

    binary_confusion_matrix = BinaryConfusionMatrix(y_actu, y_pred)
    print("Binary confusion matrix:\n%s" % binary_confusion_matrix)

It display as a nicely labeled Pandas DataFrame

::

    Binary confusion matrix:
    Predicted  False  True  __all__
    Actual
    False         67     0       67
    True          21    24       45
    __all__       88    24      112

You can get useful attributes such as True Positive (TP), True Negative
(TN) ...

::

    print binary_confusion_matrix.TP

Matplotlib plot of a binary confusion matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    binary_confusion_matrix.plot()
    plt.show()

.. figure:: screenshots/binary_cm.png
   :alt: binary\_confusion\_matrix

   binary\_confusion\_matrix

Matplotlib plot of a normalized binary confusion matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    binary_confusion_matrix.plot(normalized=True)
    plt.show()

.. figure:: screenshots/binary_cm_norm.png
   :alt: binary\_confusion\_matrix\_norm

   binary\_confusion\_matrix\_norm

Seaborn plot of a binary confusion matrix (ToDo)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    from pandas_confusion import Backend
    binary_confusion_matrix.plot(backend=Backend.Seaborn)

Confusion matrix and class statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Overall statistics and class statistics of confusion matrix can be
easily displayed.

::

    y_true = [600, 200, 200, 200, 200, 200, 200, 200, 500, 500, 500, 200, 200, 200, 200, 200, 200, 200, 200, 200]
    y_pred = [100, 200, 200, 100, 100, 200, 200, 200, 100, 200, 500, 100, 100, 100, 100, 100, 100, 100, 500, 200]
    cm = ConfusionMatrix(y_true, y_pred)
    cm.print_stats()

You should get:

::

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

Statistics are also available as an OrderedDict using:

::

    cm.stats()

Install
-------

::

    $ conda install pandas scikit-learn scipy

    $ pip install pandas_confusion

Development
-----------

You can help to develop this library.

Issues
~~~~~~

You can submit issues using
https://github.com/scls19fr/pandas_confusion/issues

Clone
~~~~~

You can clone repository to try to fix issues yourself using:

::

    $ git clone https://github.com/scls19fr/pandas_confusion.git

Run unit tests
~~~~~~~~~~~~~~

Run all unit tests

::

    $ nosetests -s -v

Run a given test

::

    $ nosetests -s -v tests/test_pandas_confusion.py:test_pandas_confusion_normalized

Install development version
~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    $ python setup.py install

or

::

    $ sudo pip install git+git://github.com/scls19fr/pandas_confusion.git

Collaborating
~~~~~~~~~~~~~

-  Fork repository
-  Create a branch which fix a given issue
-  Submit pull requests

https://help.github.com/categories/collaborating/

Done
----

-  Continuous integration (Travis)

-  Convert a confusion matrix to a binary confusion matrix

-  Python package

-  Unit tests (nose)

-  Fix missing column and missing row

-  Overall statistics: Accuracy, 95% CI, P-Value [Acc > NIR], Kappa

.. |Latest Version| image:: https://img.shields.io/pypi/v/pandas_confusion.svg
   :target: https://pypi.python.org/pypi/pandas_confusion/
.. |Supported Python versions| image:: https://img.shields.io/pypi/pyversions/pandas_confusion.svg
   :target: https://pypi.python.org/pypi/pandas_confusion/
.. |Wheel format| image:: https://img.shields.io/pypi/wheel/pandas_confusion.svg
   :target: https://pypi.python.org/pypi/pandas_confusion/
.. |License| image:: https://img.shields.io/pypi/l/pandas_confusion.svg
   :target: https://pypi.python.org/pypi/pandas_confusion/
.. |Development Status| image:: https://img.shields.io/pypi/status/pandas_confusion.svg
   :target: https://pypi.python.org/pypi/pandas_confusion/
.. |Downloads monthly| image:: https://img.shields.io/pypi/dm/pandas_confusion.svg
   :target: https://pypi.python.org/pypi/pandas_confusion/
.. |Requirements Status| image:: https://requires.io/github/scls19fr/pandas_confusion/requirements.svg?branch=master
   :target: https://requires.io/github/scls19fr/pandas_confusion/requirements/?branch=master
.. |Code Health| image:: https://landscape.io/github/scls19fr/pandas_confusion/master/landscape.svg?style=flat
   :target: https://landscape.io/github/scls19fr/pandas_confusion/master
.. |Codacy Badge| image:: https://www.codacy.com/project/badge/87be7082d9504db59d397b5738dbf133
   :target: https://www.codacy.com/app/s-celles/pandas_confusion
.. |Build Status| image:: https://travis-ci.org/scls19fr/pandas_confusion.svg
   :target: https://travis-ci.org/scls19fr/pandas_confusion
