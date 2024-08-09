import warnings

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import NotFittedError
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.validation import check_is_fitted

from mapie.control_risk.p_values import compute_hoeffdding_bentkus_p_value


class AnomalyControl:
    """
    Anomaly Detection with Optimal Threshold Selection

    Most anomaly detectors use a threshold to determine whether an input is anomalous or "normal" data.
    This class identifies the optimal threshold that meets the user's specified risk tolerance and error level.
    The "Learn and Test" principle is employed to achieve this.
    Currently, this technique is specifically applied to a deep isolation forest using the IsolationForest implementation from scikit-learn.
    In the future, small modifications may be necessary to make this class compatible with other types of anomaly detectors.


    Parameters
    ----------
    ano_detector: IsolationForest
        An anomaly detector with scikit-learn API
        (i.e. with fit, predict, and predict_proba methods), by default None.

    alpha: float between 0 and 1
        Risk tolerance.
        By default 0.2.

    delta: float between 0 and 1
        Error level.
        By default 0.01.


    Attribute
    ----------
    ano_th_: Optimal threshold

    References
    ----------
    [1] Angelopoulos, A. N., Bates, S., Candès, E. J., Jordan,
    M. I., & Lei, L. (2021). Learn then test:
    "Calibrating predictive algorithms to achieve risk control".

    """

    fit_attributes = ["ano_th_"]
    ths = np.linspace(0, 1, 100)

    def __init__(self, ano_detector, alpha: float, delta: float) -> None:
        self.ano_detector = self._check_ano_detector(ano_detector)
        self.alpha = self._check_alpha_delta(alpha)
        self.delta = self._check_alpha_delta(delta)

    def _check_ano_detector(self, ano_detector):
        if isinstance(ano_detector, IsolationForest):
            return ano_detector
        else:
            raise ValueError("The anomaly detector is not an Isolation Forest")

    def _check_alpha_delta(self, alpha: float) -> float:
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
            raise ValueError(
                "Invalid alpha/delta. Allowed values are only floats between 0 and 1"
            )
        if alpha < 0 or alpha > 1:
            raise ValueError(
                "Invalid alpha/delta. Allowed values are only floats between 0 and 1"
            )
        return alpha

    def _check_ano_detector_fitted(self) -> None:
        """
        Check if the anomaly detector is fitted to the training dataset

        Raises
        ------
        ValueError
            If the anomaly detector is not fitted to the training dataset.
        """
        try:
            check_is_fitted(self.ano_detector)
        except NotFittedError as e:
            raise ValueError(f"The anomaly detector is not fitted: {repr(e)}")

    def fit_ano_detector(self, X_train: np.array) -> "AnomalyControl":
        """
        Fit the anomaly detector chosen to the training dataset

        :param X_train: Training dataset can be the embedding of X_train

        :return:
            AnomalyControl
        """
        # check si on doit prendre embedding ? ou on dit que ça était fait avant ?
        self.ano_detector = self.ano_detector.fit(X_train)
        return self

    def fit_calibrator(self, X_calib: np.array, y_calib: np.array) -> "AnomalyControl":
        """
        Find the best threshold for the anomaly detector with respect to alpha and delta decided by the user.

        :param X_calib: Calibration dataset. Can be the embedding of X_calib.
        :param y_calib: Calibration ground truth. Only time when annotations are needed. Binary classification : Anomaly or not.
        :return:
            AnomalyControl
        """
        # check if ood_detector is fitted
        self._check_ano_detector_fitted()
        # Raise Warning if alpha is to low

        # on suppose aussi qu'on a déjà des embedding si c'est necessaire
        clf = self.ano_detector

        precisions = []
        recalls = []

        for th in self.ths:
            outlier_score_predictions = clf.decision_function(X_calib)
            y_pred = (outlier_score_predictions < th).astype(int)
            precisions.append(precision_score(y_calib, y_pred))
            recalls.append(recall_score(y_calib, y_pred))

        p_values = compute_hoeffdding_bentkus_p_value(
            1 - np.array(precisions), len(y_calib), self.alpha
        )
        valid_index = [
            i
            for i in range(len(self.ths))
            if p_values[i] <= (self.delta / len(self.ths))
        ]
        valid_ths = self.ths[valid_index]
        if len(valid_ths) > 0:
            self.ano_th_ = valid_ths.max()
        else:
            warnings.warn(
                "Alpha is too low; no valid threshold can be found with this alpha. Consider using a higher alpha value."
            )
            self.ano_th_ = self.ths.max()

        return self

    def fit(
        self, X_train: np.array, X_calib: np.array, y_calib: np.array
    ) -> "AnomalyControl":
        """
        Fit the anomaly detector to the training dataset and calibrate it (find the best threshold).

        :param X_train: training dataset
        :param X_calib: calibration dataset
        :param y_calib: calibration ground truth
        :return:
            AnomalyControl
        """
        self.fit_ano_detector(X_train)
        self.fit_calibrator(X_calib, y_calib)
        return self

    def predict(self, X: np.array) -> np.array(bool):
        """
        Differenciate anomaly from "normal" data

        :X: data with potential anomalies
        :return: boolean with True if the image is "normal" and False if it is likely to be an anomaly.

        """
        check_is_fitted(self, self.fit_attributes)

        ano_scores = self.ano_detector.predict(X)
        return ano_scores > self.ano_th_
