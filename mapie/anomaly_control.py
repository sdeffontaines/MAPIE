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

    def __init__(self, ano_detector, alpha, delta):
        self.ano_detector = ano_detector
        self.alpha = self.check_alpha_delta(alpha)
        self.delta = self.check_alpha_delta(delta)
        self.ano_th = 0

    def check_alpha_delta(self, alpha: float):
        """
        Check alpha.

        Parameters
        ----------
        alpha: float
            Can be only a float
            Between 0 and 1, represent the uncertainty of the confidence interval.
            Lower alpha produce larger (more conservative) prediction intervals.
            alpha is the complement of the target coverage level.
            By default 0.1.

        Returns
        -------
        float
            Alpha.

        Raises
        ------
        ValueError
            If alpha is not a float between 0 and 1.
        """
        if not isinstance(alpha, float):
            raise ValueError("Invalid alpha/delta. Allowed values are only one float")
        if alpha < 0 or alpha > 1:
            raise ValueError("Invalid alpha/delta. Allowed values are between 0 and 1.")
        return alpha

    def _check_parameters(self) -> None:
        """
        Perform several checks on input parameters.

        Raises
        ------
        ValueError
            If parameters are not valid.
        """
        self.check_alpha_delta(self.delta)
        self.check_alpha_delta(self.delta)

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
        self._check_parameters()
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
        # lever erreur si pas de seuil
        self.ano_th = valid_ths.max() if len(valid_ths) > 0 else 999
        return self

    def fit(self, X_train, X_calib, y_calib):
        self.fit_ano_detector(X_train)
        self.fit_calibrator(X_calib, y_calib)
        return self

    def predict(self, X):

        # check is fitted

        ano_scores = self.ano_detector.predict(X)
        return ano_scores > self.ano_th
