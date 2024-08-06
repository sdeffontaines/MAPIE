import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from mapie.control_risk.p_values import compute_hoeffdding_bentkus_p_value


class AnomalyControl:

    fit_attributes = ["ano_th"]

    def __init__(self, ano_detector, alpha, delta, nb_iter):
        self.ano_detector = ano_detector
        self.alpha = alpha
        self.delta = delta
        self.nb_iter = nb_iter

    def fit_ano_detector(self, X_train):
        """
        Fit the anomaly detector chosen to the training dataset

        :param X: _description_
        :param y: _description_
        :return: _description_
        """
        # check si on doit prendre embedding ? ou on dit que ça était fait avant ?
        self.ano_detector = self.ano_detector.fit(X_train)
        return self

    def fit_calibrator(self, X_calib, y_calib):
        # check if ood_detector is fitted
        # check alpha and delta
        # Raise Warning if alpha is to low

        # on suppose aussi qu'on a déjà des embedding si c'est necessaire
        ths = np.linspace(0, 1, 100)
        clf = self.ano_detector

        precisions = []
        recalls = []

        for th in ths:
            outlier_score_predictions = clf.decision_function(X_calib)
            y_pred = (outlier_score_predictions < th).astype(int)
            precisions.append(precision_score(y_calib, y_pred))
            recalls.append(recall_score(y_calib, y_pred))

        p_values = compute_hoeffdding_bentkus_p_value(
            1 - np.array(precisions), len(y_calib), self.alpha
        )
        valid_index = [
            i for i in range(len(ths)) if p_values[i] <= (self.delta / len(ths))
        ]
        valid_ths = ths[valid_index]
        self.ano_th = valid_ths.max()
        return self

    def fit(self, X, y, calib_size: float):
        X_train, X_calib, _, y_calib = train_test_split(X, y, test_size=calib_size)
        self.fit_ano_detector(X_train)
        self.fit_calibrator(X_calib, y_calib)
        return self

    def predict(self, X):

        # check is fitted

        ano_scores = self.ano_detector.predict(X)
        return ano_scores > self.ano_th
