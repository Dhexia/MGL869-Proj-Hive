import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from typing import Dict, Callable, Tuple
from sklearn.base import BaseEstimator


# 1. Chargement et nettoyage des données
def load_and_prepare_data(
    file_path: str,
    target_column: str,
    drop_columns: list = None,
    test_size: float = 0.3,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Charge les données, nettoie et prépare les ensembles d'entraînement et de test.
    
    :param file_path: Chemin vers le fichier CSV.
    :param target_column: Colonne cible à prédire.
    :param drop_columns: Liste des colonnes à supprimer.
    :param test_size: Proportion des données utilisées pour le test.
    :param random_state: Seed pour la reproductibilité.
    :return: X_train, X_test, y_train, y_test
    """
    # Charger les données
    df = pd.read_csv(file_path)

    # Supprimer les colonnes inutiles
    if drop_columns:
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])
    
    # Séparer les features (X) et la cible (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


# 2. Construction et entraînement du modèle
def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> BaseEstimator:
    """
    Entraîne un modèle donné sur les données.
    
    :param model: Instance de modèle à entraîner (doit respecter l'interface sklearn).
    :param X_train: Données d'entraînement (features).
    :param y_train: Labels d'entraînement.
    :return: Modèle entraîné.
    """
    model.fit(X_train, y_train)
    return model


# 3. Évaluation des performances
def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """
    Évalue les performances d'un modèle sur les données de test.
    
    :param model: Modèle à évaluer.
    :param X_test: Données de test (features).
    :param y_test: Labels de test.
    :return: Dictionnaire des métriques (AUC, précision, rappel).
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "AUC": roc_auc_score(y_test, y_proba),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred)
    }

    return metrics


# 4. Pipeline principal
def run_pipeline(
    file_path: str,
    model: Callable[[], BaseEstimator],
    target_column: str = "BugStatus",
    drop_columns: list = None,
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Exécute le pipeline complet pour entraîner et évaluer un modèle.
    
    :param file_path: Chemin vers le fichier CSV.
    :param model: Fonction ou instance qui retourne un modèle sklearn.
    :param target_column: Colonne cible à prédire.
    :param drop_columns: Liste des colonnes à supprimer.
    :param test_size: Proportion des données utilisées pour le test.
    :param random_state: Seed pour la reproductibilité.
    :return: Dictionnaire des métriques d'évaluation.
    """
    # 1. Préparer les données
    X_train, X_test, y_train, y_test = load_and_prepare_data(
        file_path, target_column, drop_columns, test_size, random_state
    )
    
    # 2. Construire et entraîner le modèle
    model_instance = model()
    trained_model = train_model(model_instance, X_train, y_train)
    
    # 3. Évaluer le modèle
    metrics = evaluate_model(trained_model, X_test, y_test)

    return metrics
