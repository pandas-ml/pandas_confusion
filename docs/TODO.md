# ToDo list

* Better documentation

* Doctest

* Matplotlib discrete colorbar (not for normalized plot)

see ColorbarBase

http://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar

* Display numbers inside cells like http://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python

* Compare with results from Sklearn

Example:

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

R like method ? `conf_mat_tab <- table(lapply(df, factor, levels = seq(100, 600, 100)))`

http://pandas.pydata.org/pandas-docs/stable/comparison_with_r.html

    idx_new_cls = pd.Index([300, 400])
    new_idx = df.index | idx_new_cls
    new_idx.name = 'Actual'
    new_col = df.index | idx_new_cls
    new_col.name = 'Predicted'
    df = df.loc[new_idx, new_col].fillna(0)


see `cm.enlarge(...)`

* Calculate Mcnemar's Test P-Value with binary confusion matrix

R code

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
