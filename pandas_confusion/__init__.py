#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Binary confusion matrix
"""

import math
import pandas as pd

class BinaryConfusionMatrix:
    """Binary confusion matrix"""
    
    def __init__(self, y_actu, y_pred):
        
        if isinstance(y_actu, pd.Series):
            self.y_actu = y_actu
        else:
            self.y_actu = pd.Series(y_actu)
            
        if isinstance(y_pred, pd.Series):
            self.y_pred = y_pred
        else:
            self.y_pred = pd.Series(y_pred)
        
        self.df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'])
        #self.df_confusion.index.name = 'Actual'
        #self.df_confusion.columns.name = 'Predicted'

        self.df_conf_norm = self.df_confusion / self.df_confusion.sum(axis=1)

    @property
    def P(self):
        """Positive"""
        return(self.df_confusion.loc[True, :].sum())
    
    @property
    def N(self):
        """Negative"""
        return(self.df_confusion.loc[False, :].sum())
    
    @property
    def TP(self):
        """
        true positive (TP)
        eqv. with hit
        """
        return(self.df_confusion.loc[True, True])
    
    @property
    def TN(self):
        """
        true negative (TN)
        eqv. with correct rejection
        """
        return(self.df_confusion.loc[False, False])
    
    @property
    def FN(self):
        """
        false negative (FN)
        eqv. with miss, Type II error
        """
        return(self.df_confusion.loc[True, False])
    
    @property
    def FP(self):
        """
        false positive (FP)
        eqv. with false alarm, Type I error
        """
        return(self.df_confusion.loc[False, True])
    
    @property
    def FPR(self):
        """
        fall-out or false positive rate (FPR)
        \mathit{FPR} = \mathit{FP} / N = \mathit{FP} / (\mathit{FP} + \mathit{TN})
        """
        #return(float(self.FP)/(self.FP + self.TN))
        return(float(self.FP) / self.N)
    
    @property
    def TPR(self):
        """
        sensitivity or true positive rate (TPR)
        eqv. with hit rate, recall
        \mathit{TPR} = \mathit{TP} / P = \mathit{TP} / (\mathit{TP}+\mathit{FN})
        """
        #return(float(self.TP) / (self.TP + self.FN))
        return(float(self.TP) / self.P)

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
        \mathit{SPC} = \mathit{TN} / N = \mathit{TN} / (\mathit{FP} + \mathit{TN}) 
        """
        return(float(self.TN) / self.N)

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
        precision or positive predictive value (PPV)
        \mathit{PPV} = \mathit{TP} / (\mathit{TP} + \mathit{FP})
        """
        return(float(self.TP) / (self.TP + self.FP))

    @property
    def precision(self):
        """
        same as PPV
        """
        return(self.PPV)

    @property
    def NPV(self):
        """
        negative predictive value (NPV)
        \mathit{NPV} = \mathit{TN} / (\mathit{TN} + \mathit{FN})
        """
        return(float(self.TN) / (self.TN + self.FN))
    
    @property
    def FDR(self):
        """
        false discovery rate (FDR)
        \mathit{FDR} = \mathit{FP} / (\mathit{FP} + \mathit{TP}) = 1 - \mathit{PPV} 
        """
        return(1 - self.PPV)
    
    @property
    def FNR(self):
        """
        Miss Rate or False Negative Rate (FNR)
        \mathit{FNR} = \mathit{FN} / P = \mathit{FN} / (\mathit{FN} + \mathit{TP}) 
        """
        return(float(self.FN) / self.P)
    
    @property
    def ACC(self):
        """
        accuracy (ACC)
        \mathit{ACC} = (\mathit{TP} + \mathit{TN}) / (P + N)
        """
        return(float(self.TP + self.TN) / (self.P + self.N))
    
    @property
    def F1_score(self):
        """
        F1 score is the harmonic mean of precision and sensitivity
        \mathit{F1} = 2 \mathit{TP} / (2 \mathit{TP} + \mathit{FP} + \mathit{FN})
        """
        return(2 * float(self.TP)/(2 * self.TP + self.FP + self.FN))
    
    @property
    def MCC(self):
        """
        Matthews correlation coefficient (MCC)
        \frac{ TP \times TN - FP \times FN } {\sqrt{ (TP+FP) ( TP + FN ) ( TN + FP ) ( TN + FN ) } }
        """
        return((self.TP * self.TN - self.FP * self.FN) \
            / math.sqrt((self.TP + self.FP) * ( self.TP + self.FN ) *\
            ( self.TN + self.FP ) * ( self.TN + self.FN )))
    
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
    
    def __repr__(self):
        return(self.df_confusion.__repr__())

    def __str__(self):
        return(self.df_confusion.__str__())

    def plot(self, **kwargs):
        import seaborn as sns
        sns.heatmap(self.df_confusion, **kwargs)
