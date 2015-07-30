#!/usr/bin/python
# -*- coding: utf8 -*-

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

"""
A Python Pandas Confusion matrix implementation
"""

import math
import numpy as np
import pandas as pd
from enum import Enum, IntEnum  # pip install enum34
import matplotlib as mpl
import matplotlib.pylab as plt
import collections

from .stats import binom_interval, class_agreement, prop_test


class Backend(Enum):
    Matplotlib = 1
    Seaborn = 2


class Axis(IntEnum):
    Actual = 1
    Predicted = 0


BACKEND_DEFAULT = Backend.Matplotlib
SUM_NAME_DEFAULT = '__all__'
DISPLAY_SUM_DEFAULT = True
TRUE_NAME_DEFAULT = 'Actual'
PRED_NAME_DEFAULT = 'Predicted'
CLASSES_NAME_DEFAULT = 'Classes'
COLORBAR_TRIG = 10

class ConfusionMatrixAbstract(object):
    """
    Abstract class for confusion matrix
    
    You shouldn't instantiate this class.
    You might instantiate ConfusionMatrix or BinaryConfusionMatrix classes
    """

    def __init__(self, y_true, y_pred, labels=None,
            display_sum=DISPLAY_SUM_DEFAULT, backend=BACKEND_DEFAULT,
            true_name = TRUE_NAME_DEFAULT, pred_name = PRED_NAME_DEFAULT):

        self.true_name = true_name
        self.pred_name = pred_name

        if isinstance(y_true, pd.Series):
            self._y_true = y_true
            self._y_true.name = self.true_name
        else:
            self._y_true = pd.Series(y_true, name=self.true_name)
        
        if isinstance(y_pred, pd.Series):
            self._y_pred = y_pred
            self._y_pred.name = self.pred_name
        else:
            self._y_pred = pd.Series(y_pred, name=self.pred_name)


        if labels is not None:
            if not self.is_binary:
                self._y_true = self._y_true.map(lambda i: self._label(i, labels))
                self._y_pred = self._y_pred.map(lambda i: self._label(i, labels))
            else:
                N = len(labels)
                assert len(labels)==2, "labels be a list with length=2 - length=%d" % N
                d = {labels[0]: False, labels[1]: True}
                self._y_true = self._y_true.map(d)
                self._y_pred = self._y_pred.map(d)
                raise(NotImplementedError) # ToDo: see self.classes and BinaryConfusionMatrix.__class ...
        
        N_true = len(y_true)
        N_pred = len(y_pred)
        assert N_true == N_pred, \
            "y_true must have same size - %d != %d" % (N_true, N_pred)

        #from sklearn.metrics import confusion_matrix
        #a = confusion_matrix(y_true, y_pred, labels=labels)
        #print(a)
        #self._df_confusion = pd.DataFrame(a, index=labels, columns=labels)
        #self._df_confusion.index.name = self.true_name
        #self._df_confusion.columns.name = self.pred_name

        #df = pd.crosstab(self._y_true, self._y_pred,
        #   rownames=[self.true_name], colnames=[self.pred_name])
        #df = pd.crosstab(self._y_true, self._y_pred,
        #    rownames=self.true_name, colnames=self.pred_name)
        df = pd.crosstab(self._y_true, self._y_pred)
        idx = self._classes(df)
        df = df.loc[idx, idx.copy()].fillna(0) # if some columns or rows are missing
        self._df_confusion = df
        self._df_confusion.index.name = self.true_name
        self._df_confusion.columns.name = self.pred_name
        self._df_confusion = self._df_confusion.astype(np.int64)

        self._len = len(idx)

        self.backend = backend
        self.display_sum = display_sum

    def _label(self, i, labels):
        try:
            return(labels[i])
        except:
            return(i)

    def __repr__(self):
        return(self.to_dataframe(calc_sum=self.display_sum).__repr__())

    def __str__(self):
        return(self.to_dataframe(calc_sum=self.display_sum).__str__())
        #return("%s:\n%s" % (self.title, self.to_dataframe(calc_sum=self.display_sum).__str__()))

    @property
    def classes(self):
        """
        Returns classes (property)
        """
        return(self._classes())

    def _classes(self, df=None):
        """
        Returns classes (method)
        """
        if df is None:
            df = self.to_dataframe()
        idx_classes = (df.columns | df.index).copy()
        idx_classes.name = CLASSES_NAME_DEFAULT
        return(idx_classes)        

    def to_dataframe(self, normalized=False, calc_sum=False, sum_label=SUM_NAME_DEFAULT):
        """
        Returns a Pandas DataFrame
        """
        if normalized:
            a = self._df_confusion.values.astype('float')
            a = a.astype('float') / a.sum(axis=1)[:, np.newaxis]
            df = pd.DataFrame(a,
                index=self._df_confusion.index.copy(),
                columns=self._df_confusion.columns.copy()
            )
        else:
            df = self._df_confusion

        if calc_sum:
            df = df.copy()
            df[sum_label] = df.sum(axis=1)
            #df = pd.concat([df, pd.DataFrame(df.sum(axis=1), columns=[sum_label])], axis=1)
            df = pd.concat([df, pd.DataFrame(df.sum(axis=0), columns=[sum_label]).T])
            df.index.name = self.true_name
        
        return(df)

    @property
    def true(self):
        """
        Returns sum of actual (true) values for each class
        """
        s = self.to_dataframe().sum(axis=1)
        s.name = self.true_name
        return(s)

    @property
    def pred(self):
        """
        Returns sum of predicted values for each class
        """
        s = self.to_dataframe().sum(axis=0)
        s.name = self.pred_name
        return(s)

    def to_array(self, normalized=False, sum=False, sum_label=SUM_NAME_DEFAULT):
        """
        Returns a Numpy Array
        """
        return(self.to_dataframe(normalized, sum, sum_label).values)

    def toarray(self, *args, **kwargs):
        """
        see to_array
        """
        return(self.to_array(*args, **kwargs))

    #@property
    def len(self):
        """
        Returns len of a confusion matrix.
        For example: 3 means that this is a 3x3 (3 rows, 3 columns) matrix
        """
        return(self._len)

    #@property
    def sum(self):
        """
        Returns sum of a confusion matrix.
        Also called "population"
        It should be the number of elements of either y_true or y_pred
        """
        return(self.to_dataframe().sum().sum())

    @property
    def population(self):
        """
        see also sum
        """
        return(self.sum())

    #@property
    def y_true(self, func=None): # Not a property (because we will add parameter)
        if func is None:
            return(self._y_true)
        else:
            return(self._y_true.map(func))

    #@property
    def y_pred(self, func=None): # Not a property (because we will add parameter)
        if func is None:
            return(self._y_pred)
        else:
            return(self._y_pred.map(func))

    @property
    def title(self):
        """
        Returns title
        """
        if self.is_binary:
            return("Binary confusion matrix")
        else:
            return("Confusion matrix")

    def plot(self, normalized=False, backend=None, ax=None, **kwargs):
        """
        Plots confusion matrix
        """
        df = self.to_dataframe(normalized)

        try:
            cmap = kwargs['cmap']
        except:
            cmap = plt.cm.gray_r

        title = self.title

        if normalized:
            title += " (normalized)"

        if backend is None:
            backend = self.backend

        if backend == Backend.Matplotlib:
            #if ax is None:
            fig, ax = plt.subplots(figsize=(9, 8))
            plt.imshow(df, cmap=cmap, interpolation='nearest') # imshow / matshow
            ax.set_title(title)

            tick_marks_col = np.arange(len(df.columns))
            tick_marks_idx = tick_marks_col.copy()

            ax.set_yticks(tick_marks_idx)
            ax.set_xticks(tick_marks_col)
            ax.set_xticklabels(df.columns, rotation=45, ha='right')
            ax.set_yticklabels(df.index)
            
            ax.set_ylabel(df.index.name)
            ax.set_xlabel(df.columns.name)

            (N_min, N_max) = (0, self.max())
            if N_max > COLORBAR_TRIG:
                plt.colorbar() # Continuous colorbar
            else:
                # Discrete colorbar
                ax2 = fig.add_axes([0.93, 0.1, 0.03, 0.8])
                bounds = np.arange(N_min, N_max + 2, 1)
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                cb = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')

            return(ax)


        elif backend == Backend.Seaborn:
            import seaborn as sns
            ax = sns.heatmap(df, **kwargs)
            return(ax)
            # You should test this yourself
            # because I'm facing an issue with Seaborn under Mac OS X (2015-04-26)
            # RuntimeError: Cannot get window extent w/o renderer
            #sns.plt.show()

        else:
            raise(NotImplementedError("backend=%r not allowed" % backend))

    def binarize(self, select):
        """Returns a binary confusion matrix from
        a confusion matrix"""
        if not isinstance(select, collections.Iterable):
            select = np.array(select)

        y_true_bin = self.y_true().map(lambda x: x in select)
        y_pred_bin = self.y_pred().map(lambda x: x in select)

        binary_cm = BinaryConfusionMatrix(y_true_bin, y_pred_bin)

        return(binary_cm)

    def enlarge(self, select):
        """
        Enlarges confusion matrix with new classes
        It should add empty rows and columns
        """
        if not isinstance(select, collections.Iterable):
            idx_new_cls = pd.Index([select])
        else:
            idx_new_cls = pd.Index(select)
        new_idx = self._df_confusion.index | idx_new_cls
        new_idx.name = self.true_name
        new_col = self._df_confusion.columns | idx_new_cls
        new_col.name = self.pred_name
        print(new_col)
        self._df_confusion = self._df_confusion.loc[:, new_col]
        #self._df_confusion = self._df_confusion.loc[new_idx, new_col].fillna(0)
        #ToFix: KeyError: 'the label [True] is not in the [index]'

    @property
    def stats_overall(self):
        """
        Returns an OrderedDict with overall statistics
        """
        df = self._df_confusion
        d_stats = collections.OrderedDict()

        d_class_agreement = class_agreement(df)

        key = 'Accuracy'
        try:
            d_stats[key] = d_class_agreement['diag']  #0.35
        except:
            d_stats[key] = np.nan

        key = '95% CI'
        try:
            d_stats[key] = binom_interval(np.sum(np.diag(df)), df.sum().sum())  #(0.1539, 0.5922)
        except:
            d_stats[key] = np.nan
        
        d_prop_test = prop_test(df)
        d_stats['No Information Rate'] = 'ToDo'  #0.8

        d_stats['P-Value [Acc > NIR]'] = d_prop_test['p.value']  #1

        d_stats['Kappa'] = d_class_agreement['kappa']  #0.078

        d_stats['Mcnemar\'s Test P-Value'] = 'ToDo'  #np.nan

        return(d_stats)

    @property
    def stats_class(self):
        """
        Returns a DataFrame with class statistics
        """
        #stats = ['TN', 'FP', 'FN', 'TP']
        #df = pd.DataFrame(columns=self.classes, index=stats)
        df = pd.DataFrame(columns=self.classes)

        # ToDo Avoid these for loops

        for cls in self.classes:
            binary_cm = self.binarize(cls)
            binary_cm_stats = binary_cm.stats()
            for key, value in binary_cm_stats.items():
                df.loc[key, cls] = value #binary_cm_stats

        d_name = {
            'population': 'Population',
            'P': 'P: Condition positive',
            'N': 'N: Condition negative',
            'PositiveTest': 'Test outcome positive',
            'NegativeTest': 'Test outcome negative',
            'TP': 'TP: True Positive',
            'TN': 'TN: True Negative',
            'FP': 'FP: False Positive',
            'FN': 'FN: False Negative',
            'TPR': 'TPR: (Sensitivity, hit rate, recall)', # True Positive Rate 
            'TNR': 'TNR=SPC: (Specificity)', # True Negative Rate 
            'PPV': 'PPV: Pos Pred Value (Precision)',
            'NPV': 'NPV: Neg Pred Value',
            'prevalence': 'Prevalence',
            #'xxx': 'xxx: Detection Rate',
            #'xxx': 'xxx: Detection Prevalence',
            #'xxx': 'xxx: Balanced Accuracy',
            'FPR': 'FPR: False-out',
            'FDR': 'FDR: False Discovery Rate',
            'FNR': 'FNR: Miss Rate',
            'ACC': 'ACC: Accuracy',
            'F1_score': 'F1 score',
            'MCC': 'MCC: Matthews correlation coefficient',
            'informedness': 'Informedness',
            'markedness': 'Markedness',
            'LRP': 'LR+: Positive likelihood ratio',
            'LRN': 'LR-: Negative likelihood ratio',
            'DOR': 'DOR: Diagnostic odds ratio',
            'FOR': 'FOR: False omission rate',
        }
        df.index = df.index.map(lambda id: self._name_from_dict(id, d_name))

        return(df)

    def stats(self, lst_stats=None):
        """
        Return an OrderedDict with statistics
        """
        d_stats = collections.OrderedDict()
        d_stats['cm'] = self
        d_stats['overall'] = self.stats_overall
        d_stats['class'] = self.stats_class
        return(d_stats)

    def _name_from_dict(self, key, d_name):
        """
        Returns name (value in dict d_name
        or key if key doesn't exists in d_name)
        """
        try:
            return(d_name[key])
        except:
            return(key)

    def _str_dict(self, d, line_feed_key_val='\n', 
            line_feed_stats='\n\n', d_name=None):
        """
        Return a string representation of a dictionary
        """
        s = ""
        for i, (key, val) in enumerate(d.items()):
            name = self._name_from_dict(key, d_name)
            if i != 0:
                s = s + line_feed_stats
            s = s + "%s:%s%s" % (name, line_feed_key_val, val)
        return(s)   

    def _str_stats(self, lst_stats=None):
        """
        Returns a string representation of statistics
        """
        d_stats_name = {
            "cm": "Confusion Matrix",
            "overall": "Overall Statistics",
            "class": "Class Statistics",
        }

        stats = self.stats(lst_stats)

        d_stats_str = collections.OrderedDict([
            ("cm", str(stats['cm'])),
            ("overall", self._str_dict(stats['overall'],
                line_feed_key_val=' ', line_feed_stats='\n')),
            ("class", str(stats['class'])),
        ])

        s = self._str_dict(d_stats_str, line_feed_key_val='\n\n',
            line_feed_stats='\n\n\n', d_name=d_stats_name)
        return(s)

    def print_stats(self, lst_stats=None):
        """
        Prints statistics
        """
        print(self._str_stats(lst_stats))

    def get(self, actual=None, predicted=None):
        """
        Get confusion matrix value for a given
        actual class and a given predicted class

        if only one parameter is given (actual or predicted)
        we get confusion matrix value for actual=actual and predicted=actual
        """
        if actual is None:
            actual = predicted
        if predicted is None:
            predicted = actual
        return(self.to_dataframe().loc[actual, predicted])

    #@property
    def max(self):
        """
        Returns max value of confusion matrix
        """
        return(self.to_dataframe().max().max())

    #@property
    def min(self):
        """
        Returns min value of confusion matrix
        """
        return(self.to_dataframe().min().min())

    @property
    def is_binary(self):
        """Return False"""
        return(False)

    @property
    def classification_report(self):
        """
        Returns a DataFrame with classification report
        """
        columns = np.array(['precision', 'recall', 'F1_score', 'support'])
        index = self.classes
        df = pd.DataFrame(index=index, columns=columns)

        for cls in self.classes:
            binary_cm = self.binarize(cls)
            for stat in columns:
                df.loc[cls, stat] = getattr(binary_cm, stat)

        total_support = df.support.sum()
        df.loc['__avg / total__', :] = (df[df.columns[:-1]].transpose() * df.support).sum(axis=1) / df.support.sum()
        df.loc['__avg / total__', 'support'] = total_support
        
        return(df)

    def _avg_stat(self, stat):
        """
        Binarizes confusion matrix
        and returns (weighted) average statistics
        """
        s_values = pd.Series(index = self.classes)
        for cls in self.classes:
            binary_cm = self.binarize(cls)
            v = getattr(binary_cm, stat)
            print(v)
            s_values[cls] = v
        value = (s_values * self.true).sum() / self.population
        return(value)


class ConfusionMatrix(ConfusionMatrixAbstract):
    """
    Confusion matrix class (not binary)
    """

    def __getattr__(self, attr):
        """
        Returns (weighted) average statistics
        """
        return(self._avg_stat(attr))

class BinaryConfusionMatrix(ConfusionMatrixAbstract):
    """
    Binary confusion matrix class
    """
    
    def __init__(self, *args, **kwargs):
        #super(BinaryConfusionMatrix, self).__init__(y_true, y_pred)
        super(BinaryConfusionMatrix, self).__init__(*args, **kwargs)
        assert self.len() == 2, \
            "Binary confusion matrix must have len=2 but \
len=%d because y_true.unique()=%s y_pred.unique()=%s" \
% (self.len(), self.y_true().unique(), self.y_pred().unique())

    @classmethod
    def help(cls):
        """
        Returns a DataFrame reminder about terms
        * TN: True Negative
        * FP: False Positive
        * FN: False Negative
        * TP: True Positive
        """
        df = pd.DataFrame([["TN", "FP"],["FN", "TP"]],
                columns=[False, True], index=[False, True])
        df.index.name = TRUE_NAME_DEFAULT
        df.columns.name = PRED_NAME_DEFAULT
        return(df)

    @property
    def is_binary(self):
        """Return True"""
        return(True)

    def _class(self, direction):
        """Returns class for a given direction
        direction being a boolean
        True for positive class
        False for negative class"""
        if direction:
            return(self.pos_class)
        else:
            return(self.neg_class)

    @property
    def pos_class(self):
        """Returns positive class
        If BinaryConfusionMatrix was instantiate using y_true and y_pred
        as array of booleans, it should return True
        Else it should return the name (string) of the positive class"""
        return(self.classes[1])

    @property
    def neg_class(self):
        """Returns negative class
        If BinaryConfusionMatrix was instantiate using y_true and y_pred
        as array of booleans, it should return False
        Else it should return the name (string) of the negative class"""
        return(self.classes[0])

    def dict_class(self, reversed=False):
        if not reversed:
            d = {
                self.classes[0]: False,
                self.classes[1]: True,
            }
        else:
            d = {
                False: self.classes[0],
                True: self.classes[1]
            }
        return(d)

    def y_true(self, to_bool=False):
        if not to_bool:
            return(self._y_true)
        else:
            d = self.dict_class()
            return(self._y_true.map(d))

    def y_pred(self, to_bool=False):
        if not to_bool:
            return(self._y_pred)
        else:
            d = self.dict_class()
            return(self._y_pred.map(d))

    @property
    def P(self):
        """Condition positive
        eqv. with support"""
        return(self._df_confusion.loc[self._class(True), :].sum())

    @property
    def support(self):
        """
        same as P
        """
        return(self.P)
    
    @property
    def N(self):
        """Condition negative"""
        return(self._df_confusion.loc[self._class(False), :].sum())
    
    @property
    def TP(self):
        """
        true positive (TP)
        eqv. with hit
        """
        return(self._df_confusion.loc[self._class(True), self._class(True)])

    @property
    def hit(self):
        """
        same as TP
        """
        return(self.TP)
    
    @property
    def TN(self):
        """
        true negative (TN)
        eqv. with correct rejection
        """
        return(self._df_confusion.loc[self._class(False), self._class(False)])
    
    @property
    def FN(self):
        """
        false negative (FN)
        eqv. with miss, Type II error / Type 2 error
        """
        return(self._df_confusion.loc[self._class(True), self._class(False)])
    
    @property
    def FP(self):
        """
        false positive (FP)
        eqv. with false alarm, Type I error / Type 1 error
        """
        return(self._df_confusion.loc[self._class(False), self._class(True)])

    @property
    def PositiveTest(self):
        """
        test outcome positive
        TP + FP
        """
        return(self.TP + self.FP)

    @property
    def NegativeTest(self):
        """
        test outcome negative
        TN + FN
        """
        return(self.TN + self.FN)
    
    @property
    def FPR(self):
        """
        false positive rate (FPR)
        eqv. with fall-out
        FPR = FP / N = FP / (FP + TN)
        """
        #return(np.float64(self.FP)/(self.FP + self.TN))
        return(np.float64(self.FP) / self.N)
    
    @property
    def TPR(self):
        """
        true positive rate (TPR)
        eqv. with hit rate, recall, sensitivity
        TPR = TP / P = TP / (TP+FN)
        """
        #return(np.float64(self.TP) / (self.TP + self.FN))
        return(np.float64(self.TP) / self.P)

    @property
    def recall(self):
        """
        same as TPR
        """
        return(self.TPR)


    @property
    def sensitivity(self):
        """
        same as TPR
        """
        return(self.TPR)


    @property
    def TNR(self):
        """
        specificity (SPC) or true negative rate (TNR)
        SPC = TN / N = TN / (FP + TN)
        """
        return(np.float64(self.TN) / self.N)

    @property
    def SPC(self):
        """
        same as TNR
        """
        return(self.TNR)

    @property
    def specificity(self):
        """
        same as TNR
        """
        return(self.TNR)

    @property
    def PPV(self):
        """
        positive predictive value (PPV)
        eqv. with precision
        PPV = TP / (TP + FP) = TP / PositiveTest
        """
        return(np.float64(self.TP) / self.PositiveTest)

    @property
    def precision(self):
        """
        same as PPV
        """
        return(self.PPV)

    @property
    def FOR(self):
        """
        false omission rate (FOR)
        FOR = FN / NegativeTest
        """
        return(np.float64(self.FN) / self.NegativeTest)


    @property
    def NPV(self):
        """
        negative predictive value (NPV)
        NPV = TN / (TN + FN)
        """
        return(np.float64(self.TN) / self.NegativeTest)
    
    @property
    def FDR(self):
        """
        false discovery rate (FDR)
        FDR = FP / (FP + TP) = 1 - PPV
        """
        return(np.float64(self.FP) / self.PositiveTest)
        #return(1 - self.PPV)
    
    @property
    def FNR(self):
        """
        Miss Rate or False Negative Rate (FNR)
        FNR = FN / P = FN / (FN + TP)
        """
        return(np.float64(self.FN) / self.P)
    
    @property
    def ACC(self):
        """
        accuracy (ACC)
        ACC} = (TP + TN) / (P + N) = (TP + TN) / TotalPopulation
        """
        return(np.float64(self.TP + self.TN) / self.population)
    
    @property
    def F1_score(self):
        """
        F1 score is the harmonic mean of precision and sensitivity
        F1 = 2 TP / (2 TP + FP + FN)
        can be also F1 = 2 * (precision * recall) / (precision + recall)
        """
        return(2 * np.float64(self.TP)/(2 * self.TP + self.FP + self.FN))
    
    @property
    def MCC(self):
        """
        Matthews correlation coefficient (MCC)
        \frac{ TP \times TN - FP \times FN }
             {\sqrt{ (TP+FP) ( TP + FN ) ( TN + FP ) ( TN + FN ) }
        """
        return((self.TP * self.TN - self.FP * self.FN) \
            / math.sqrt((self.TP + self.FP) * ( self.TP + self.FN ) \
            * ( self.TN + self.FP ) * ( self.TN + self.FN )))
    
    @property
    def informedness(self):
        """
        Informedness = Sensitivity + Specificity - 1
        """
        return(self.sensitivity + self.specificity - 1.0)
    
    @property
    def markedness(self):
        """
        Markedness = Precision + NPV - 1
        """
        return(self.precision + self.NPV - 1.0)

    @property
    def prevalence(self):
        """
        Prevalence = P / TotalPopulation
        """
        return(np.float64(self.P) / self.population)

    @property
    def LRP(self):
        """
        Positive likelihood ratio (LR+) = TPR / FPR
        """
        return(np.float64(self.TPR) / self.FPR)

    @property
    def LRN(self):
        """
        Negative likelihood ratio (LR-) = FNR / TNR
        """
        return(np.float64(self.FNR) / self.TNR)

    @property
    def DOR(self):
        """
        Diagnostic odds ratio (DOR) = LR+ / LR−
        """
        return(np.float64(self.LRP) / self.LRN)

    def stats(self, lst_stats=None):
        """
        Returns an ordered dictionary of statistics
        """
        if lst_stats is None:
            lst_stats = ['population', 'P', 'N', 'PositiveTest', 'NegativeTest',
                'TP', 'TN', 'FP', 'FN', 'TPR', 'TNR', 'PPV', 'NPV', 'FPR', 'FDR',
                'FNR', 'ACC', 'F1_score', 'MCC', 'informedness', 'markedness', 
                'prevalence', 'LRP', 'LRN', 'DOR', 'FOR']
        d = map(lambda stat: (stat, getattr(self, stat)), lst_stats)
        return(collections.OrderedDict(d))

    def _str_stats(self, lst_stats=None):
        """
        Returns a string representation of statistics
        """
        return(self._str_dict(self.stats(lst_stats),
            line_feed_key_val=' ', line_feed_stats='\n', d_name=None))

    def inverse(self):
        """
        Inverses a binary confusion matrix
        False -> True
        True -> False
        """
        negative_class = self.classes[0] # False
        return(self.binarize(negative_class))
