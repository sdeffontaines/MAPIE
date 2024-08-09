import os

import numpy as np
import pytest
import tensorflow as tf
from mapie.anomaly_control import AnomalyControl
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
NUMBER_OF_DISPLAYED_OUTLIERS = 50
ANOMALY_PERCENTAGE = 0.2  # Percentage of MNIST data in the train set (ici : 1%)
ALPHA = 0.2
DELTA = 0.01
NB_ITER = 10

all_precisions = []
all_recalls = []
all_p_values = []


def prep_data(anomaly_percentage: float) -> tuple:
    """Prepare the dataset for the experience. Load the dataset. Preprocess and split it.
    The normal data are images of boat from CIFAR-10 (training set) and the anomalies are handwritten digits from MNIST.

    :return:
        X_train
        X_calib
        X_test
        y_calib
        y_test
    """
    # Load data
    (X_train_anomaly, _), _ = tf.keras.datasets.mnist.load_data()
    (X_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
    X_train_boat = X_train[y_train.flatten() == 8]
    X_train = X_train_boat
    # Normalize CIFAR-10 images to [0, 1] range
    X_train = X_train / 255.0
    # Normalize MNIST images to [0, 1] range and expand dimensions to match CIFAR-10 shape
    X_train_anomaly = X_train_anomaly.astype("float32") / 255.0
    X_train_anomaly = np.expand_dims(X_train_anomaly, axis=-1)
    num_anomaly_samples = int(len(X_train) * anomaly_percentage)
    # Randomly select anomaly samples and resize them
    anomaly_images_resized = []
    for img in X_train_anomaly:
        img_resized = tf.image.resize(img, [32, 32]).numpy()
        anomaly_images_resized.append(img_resized)

    anomaly_images_resized = np.array(anomaly_images_resized)
    # Nombre total d'échantillons dans le jeu de données d'entraînement
    num_samples = len(X_train)
    # Initialiser y_train avec des 1 pour les données normales
    y = np.zeros(num_samples)
    # Indices aléatoires pour insérer les images d'anomalies
    random_indices = np.random.choice(num_samples, num_anomaly_samples, replace=False)
    # Insérer les images d'anomalies dans X_train
    X = np.copy(X_train)
    X[random_indices] = anomaly_images_resized[:num_anomaly_samples]
    # Mettre à jour les étiquettes dans y_train pour les anomalies
    y[random_indices] = 1
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_calib, y_calib


def create_embedding(X_train: np.array, X_calib: np.array) -> tuple:
    """
    Generate embeddings from a pretrained model (resnet256 pretrained on Cifar10)

    :param X_train: training dataset
    :param X_calib: calibration dataset
    :return:
        embeddings
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    model_path = os.path.expanduser("~/") + ".oodeel/saved_models"
    os.makedirs(model_path, exist_ok=True)

    model_path_resnet_cifar10 = tf.keras.utils.get_file(
        "cifar10_resnet256.h5",
        origin="https://share.deel.ai/s/kram9kLpx6JwRX4/download/cifar10_resnet256.h5",
        cache_dir=model_path,
        cache_subdir="",
    )
    model = tf.keras.models.load_model(model_path_resnet_cifar10)
    embeddings_train = model.predict(X_train)
    embeddings_calib = model.predict(X_calib)
    return embeddings_train, embeddings_calib


############### Tests #########
X_TRAIN, X_CALIB, Y_CALIB = prep_data(ANOMALY_PERCENTAGE)
EMBEDDINGS_TRAIN, EMBEDDINGS_CALIB = create_embedding(X_TRAIN, X_CALIB)


def test_initialized() -> None:
    """Test that initialization does not crash."""
    dif = IsolationForest(random_state=RANDOM_STATE, contamination=ANOMALY_PERCENTAGE)
    AnomalyControl(ano_detector=dif, alpha=ALPHA, delta=DELTA)


@pytest.mark.parametrize("alpha", [1.2, "not a float"])
def test_check_alpha(alpha: float) -> None:
    """Test that alpha has a valid value

    :param alpha: risk tolerance
    """
    dif = IsolationForest(random_state=RANDOM_STATE, contamination=ANOMALY_PERCENTAGE)
    with pytest.raises(
        ValueError,
        match="Invalid alpha/delta. Allowed values are only floats between 0 and 1",
    ):
        AnomalyControl(ano_detector=dif, alpha=alpha, delta=DELTA)


def test_check_delta() -> None:
    """Test that delta has a valid value

    :param delta: error level
    """
    dif = IsolationForest(random_state=RANDOM_STATE, contamination=ANOMALY_PERCENTAGE)
    with pytest.raises(
        ValueError,
        match="Invalid alpha/delta. Allowed values are only floats between 0 and 1",
    ):
        AnomalyControl(ano_detector=dif, alpha=ALPHA, delta=1.2)


def test_fit_ano_detector() -> None:
    """Test the fitting of the anomaly detector to the training dataset to see if it does not crash."""
    dif = IsolationForest(random_state=RANDOM_STATE, contamination=ANOMALY_PERCENTAGE)
    ano = AnomalyControl(ano_detector=dif, alpha=ALPHA, delta=DELTA)
    ano.fit_ano_detector(EMBEDDINGS_TRAIN)


def test_fit() -> None:
    """Test the global function fit."""
    dif = IsolationForest(random_state=RANDOM_STATE, contamination=ANOMALY_PERCENTAGE)
    ano = AnomalyControl(ano_detector=dif, alpha=ALPHA, delta=DELTA)
    ano.fit(EMBEDDINGS_TRAIN, EMBEDDINGS_CALIB, Y_CALIB)


def test_check_ano_detector_fitted() -> None:
    """Test"""
    dif = IsolationForest(random_state=RANDOM_STATE, contamination=ANOMALY_PERCENTAGE)
    ano = AnomalyControl(ano_detector=dif, alpha=ALPHA, delta=DELTA)
    with pytest.raises(ValueError, match="The anomaly detector is not fitted"):
        ano._check_ano_detector_fitted()


def test_predict() -> None:
    dif = IsolationForest(random_state=RANDOM_STATE, contamination=ANOMALY_PERCENTAGE)
    ano = AnomalyControl(ano_detector=dif, alpha=ALPHA, delta=DELTA)
    ano.fit(EMBEDDINGS_TRAIN, EMBEDDINGS_CALIB, Y_CALIB)
    predictions = ano.predict(EMBEDDINGS_CALIB)
    assert len(predictions) == len(EMBEDDINGS_CALIB), "Prediction length mismatch."
    assert predictions.dtype == bool, "Predictions should be boolean."


def test_predict2() -> None:
    """_summary_"""
    dif = IsolationForest(random_state=RANDOM_STATE, contamination=ANOMALY_PERCENTAGE)
    ano = AnomalyControl(ano_detector=dif, alpha=ALPHA, delta=DELTA)
    with pytest.raises(
        ValueError,
        match="This AnomalyControl instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.",
    ):
        ano.predict(EMBEDDINGS_CALIB)


def test_low_alpha_warning() -> None:
    """Test that a warning is raised for low alpha."""
    dif = IsolationForest(random_state=RANDOM_STATE, contamination=ANOMALY_PERCENTAGE)
    low_alpha = 0.00001
    ano = AnomalyControl(ano_detector=dif, alpha=low_alpha, delta=DELTA)
    ano.fit(EMBEDDINGS_TRAIN, EMBEDDINGS_CALIB, Y_CALIB)
    with pytest.warns(
        UserWarning,
        match="Alpha is too low; no valid threshold can be found with this alpha. Consider using a higher alpha value.",
    ):
        ano.fit_calibrator(X_calib=EMBEDDINGS_CALIB, y_calib=Y_CALIB)
