# pandas_confusion
 
A [Python]() [Pandas](http://pandas.pydata.org/) implementation of [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).

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

![confusion_matrix](screenshots/confusion_matrix.png)

### Matplotlib plot of a normalized confusion matrix

    confusion_matrix.plot(normalized=True)
    plt.show()

![confusion_matrix_norm](screenshots/confusion_matrix_norm.png)

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

Here is a list of attributes you can get

    TP: 24.000000
    TN: 67.000000
    FP: 0.000000
    FN: 21.000000
    TPR: 0.533333
    TNR: 1.000000
    PPV: 1.000000
    NPV: 0.761364
    FPR: 0.000000
    FDR: 0.000000
    FNR: 0.466667
    ACC: 0.812500
    F1_score: 0.695652
    MCC: 0.637229
    informedness: 0.533333
    markedness: 0.761364

### Matplotlib plot of a binary confusion matrix

    binary_confusion_matrix.plot()
    plt.show()

![binary_confusion_matrix](screenshots/binary_confusion_matrix.png)

### Matplotlib plot of a normalized binary confusion matrix

    binary_confusion_matrix.plot(normalized=True)
    plt.show()

![binary_confusion_matrix_norm](screenshots/binary_confusion_matrix_norm.png)

### Seaborn plot of a binary confusion matrix (ToDo)

    binary_confusion_matrix.plot(backend=Backend.Seaborn)

## ToDo list

* Convert a confusion matrix to a binary confusion matrix

* Display numbers inside cells like http://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python

* Python package

* Unit tests (nose)

* Compare with results from Sklearn

    from sklearn.metrics import f1_score, classification_report
    f1_score(y_actu, y_pred)
    print classification_report(y_actu, y_pred)

* Compare with R "caret" package

http://stackoverflow.com/questions/26631814/create-a-confusion-matrix-from-a-dataframe
    
* Continuous integration (Travis)

* Code metrics (landscape.io)

* Create fake truth, prediction from confusion matrix
(can be useful for unit test)

https://www.researchgate.net/post/Can_someone_help_me_to_calculate_accuracy_sensitivity_of_a_66_confusion_matrix

[see code (ToDo)](samples/fake_convol_mat.py)

## Done

* Fix missing column and missing row
