import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve
from typing import Dict, Tuple
from sklearn.base import BaseEstimator
from configparser import ConfigParser
import time
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import shap
import numpy as np
from typing import Union, Dict, Tuple



config=ConfigParser()
config.read('config.ini')

# 1. Load configuration for a specific version or scenario
def load_config(config_section: str) -> Dict[str, any]:
    """
    Loads the configuration parameters for a specific section.

    :param config_section: Section name in the config.ini file.
    :return: Dictionary containing configuration parameters.
    """
    
    if config_section not in config:
        raise ValueError(f"Section '{config_section}' not found in config.ini.")
    
    model_config = {
        "target_column": config[config_section]["target_column"],
        "drop_columns": config[config_section]["drop_columns"].split(","),
        "test_size": float(config[config_section].get("test_size", 0.3)),
        "random_state": int(config[config_section].get("random_state", 42)),
        "n_estimators": int(config[config_section].get("n_estimators", 100)),  # Default to 100 if not in config
        "max_depth": int(config[config_section].get("max_depth", None)) if config[config_section].get("max_depth") else None  # None if not in config
    }
    return model_config


# def load_config(config_section: str) -> Dict[str, any]:
#     """
#     Loads all configuration parameters for a specific section.

#     :param config_section: Section name in the config.ini file.
#     :return: Dictionary containing all configuration parameters.
#     """
#     if config_section not in config:
#         raise ValueError(f"Section '{config_section}' not found in config.ini.")

#     # Convert all keys to appropriate types (int, float, etc.)
#     model_config = {}
#     for key, value in config[config_section].items():
#         try:
#             # Try to cast to int or float if applicable
#             if '.' in value:
#                 model_config[key] = float(value)
#             else:
#                 model_config[key] = int(value)
#         except ValueError:
#             # Keep as string if casting fails
#             model_config[key] = value.strip()

#     return model_config



# 2. Load and prepare data
def load_and_prepare_data(
    data: Union[str, pd.DataFrame],
    config: Dict[str, any],
    resampling_method: str = "smote",
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads data, cleans it, balances classes, and prepares training and testing sets.
    Can accept either a file path or a preloaded DataFrame.

    Parameters:
        data (str | pd.DataFrame): Path to the CSV file or a preloaded DataFrame.
        config (dict): Configuration settings.
        resampling_method (str): Resampling strategy ("smote" or "undersample").
        verbose (bool): Whether to print detailed logs.

    Returns:
        X_train, X_test, y_train, y_test: Processed training and testing data.
    """
    # Load data if a file path is provided
    if isinstance(data, str):
        df = pd.read_csv(data, sep=",")
    elif isinstance(data, pd.DataFrame):
        df = data.copy()  # Work with a copy of the DataFrame
    else:
        raise ValueError("The 'data' parameter must be either a file path or a DataFrame.")

    # Replace commas in numerical columns with dots
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.replace(",", ".", regex=False)

    # Convert numerical columns to numeric
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass

    # Drop unused columns
    drop_columns = config.get("drop_columns", [])
    if drop_columns:
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    # Separate features (X) and target (y)
    target_column = config["target_column"]
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Normalize features
    scaler = StandardScaler()
    start_time = time.time()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    end_time = time.time()
    if not verbose:
        print(f"Scaler fit/transform time: {end_time - start_time:.2f} seconds")

    # Balance classes if requested
    if resampling_method == "undersample":
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=config.get("random_state", 42))
        X, y = sampler.fit_resample(X, y)
    elif resampling_method == "smote":
        from imblearn.over_sampling import SMOTE
        sampler = SMOTE(random_state=config.get("random_state", 42))
        X, y = sampler.fit_resample(X, y)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.get("test_size", 0.3), random_state=config.get("random_state", 42)
    )

    return X_train, X_test, y_train, y_test



# 3. Train model
def train_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    verbose: bool = False
) -> BaseEstimator:
    """
    Trains a given model on the provided data.
    """

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    if not verbose:
        print(f"Training time: {end_time - start_time:.2f} seconds")

    return model


# 4. Evaluate model
def evaluate_model(
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluates a model's performance on test data and replaces numeric class labels
    with their corresponding priority names.
    """
    # Mapping des labels
    priority_mapping = {
        0.0: "None",
        1.0: "Trivial",
        2.0: "Minor",
        3.0: "Major",
        4.0: "Critical",
        5.0: "Blocker"
    }

    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()

    if not verbose:
        print(f"Prediction time: {end_time - start_time:.2f} seconds")

    metrics = {}
    if len(np.unique(y_test)) > 2:  # Multiclass
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        y_proba = model.predict_proba(X_test)
        metrics["Accuracy"] = accuracy_score(y_test, y_pred)
        metrics["F1_macro"] = f1_score(y_test, y_pred, average="macro")
        metrics["F1_micro"] = f1_score(y_test, y_pred, average="micro")
        
        # Générer le rapport
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Mettre à jour uniquement les clés numériques
        updated_report = {}
        for k, v in report.items():
            try:
                updated_key = priority_mapping.get(float(k), k)  # Convert only numeric keys
            except ValueError:
                updated_key = k  # Keep non-numeric keys like 'accuracy'
            updated_report[updated_key] = v
        
        metrics["ClassificationReport"] = updated_report

    else:  # Binary classification
        y_proba = model.predict_proba(X_test)[:, 1]
        from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        metrics = {
            "AUC": roc_auc_score(y_test, y_proba),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "FPR": fpr.tolist(),
            "TPR": tpr.tolist()
        }

    return metrics





# 5. Plot feature importance for Random Forest
def plot_feature_importance_rf(trained_model_rf, feature_columns, top_n=30):
    """
    Plot the top N feature importances from a trained Random Forest model, 
    with a cumulative bar for all other features.

    Parameters:
        trained_model_rf (RandomForestClassifier): Trained Random Forest model.
        feature_columns (list or array-like): Column names of the dataset.
        top_n (int): Number of top features to display (default is 30).
    """
    # Extract feature importances
    feature_importances = trained_model_rf.feature_importances_

    # Associate importances with feature names
    feature_importances_df = pd.DataFrame({
        "Feature": feature_columns,  
        "Importance": feature_importances
    }).sort_values(by="Importance", ascending=False)

    # Count how many features are in "Others"
    others_count = len(feature_importances_df) - top_n

    # Select the top N features and calculate the cumulative importance of the rest
    top_features = feature_importances_df[:top_n]
    others_importance = feature_importances_df[top_n:]["Importance"].sum()
    others_row = pd.DataFrame({
        "Feature": [f"Others ({others_count} features)"], 
        "Importance": [others_importance]
    })

    # Combine top N features with the "Others" row
    final_df = pd.concat([top_features, others_row], ignore_index=True)

    # Visualization
    plt.figure(figsize=(14, 10))
    plt.barh(final_df["Feature"], final_df["Importance"])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.gca().invert_yaxis()  # Invert to show the most important feature at the top
    plt.show()



# 6. Plot SHAP summary plot
def plot_shap_summary(trained_model, X_train, X_test, top_n=10):
    """
    Generates a SHAP summary plot for the top N most important features.
    
    Parameters:
        trained_model: Trained model (e.g., Logistic Regression or any model compatible with SHAP).
        X_train: Training dataset used to fit the SHAP explainer.
        X_test: Test dataset to compute SHAP values.
        top_n: Number of top features to display in the summary plot (default is 10).
    """
    # Prepare SHAP explainer and compute SHAP values
    explainer = shap.LinearExplainer(trained_model, X_train)
    shap_values = explainer.shap_values(X_test)
    
    # Generate the summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, 
        X_test, 
        feature_names=X_train.columns, 
        max_display=top_n,  # Display top N features
        show=True
    )


def plot_shap_lr(trained_model, X_train, X_test, top_n=10):
    """
    Plot the top N features by mean SHAP values with a single "Others" bar for less important features.
    
    Parameters:
        trained_model: Trained model (e.g., Logistic Regression or any compatible model with SHAP).
        X_train: Training dataset used to fit the SHAP explainer.
        X_test: Test dataset to compute SHAP values.
        top_n: Number of top features to display (default is 10).
    """
    # Prepare SHAP explainer and compute SHAP values
    explainer = shap.LinearExplainer(trained_model, X_train)
    shap_values = explainer.shap_values(X_test)
    
    # Compute mean absolute SHAP values for each feature
    mean_shap_values = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.DataFrame({
        "Feature": X_train.columns,
        "Mean_SHAP": mean_shap_values
    }).sort_values(by="Mean_SHAP", ascending=False)
    
    # Select the top N features and calculate the cumulative importance of "Others"
    top_features = feature_importance[:top_n]
    others_count = len(feature_importance) - top_n
    others_importance = feature_importance[top_n:]["Mean_SHAP"].sum()
    
    # Create a row for "Others"
    others_row = pd.DataFrame({
        "Feature": [f"Others ({others_count} features)"],
        "Mean_SHAP": [others_importance]
    })
    
    # Combine top features with "Others"
    final_df = pd.concat([top_features, others_row], ignore_index=True)
    
    # Plot the bar chart
    plt.figure(figsize=(14, 10))
    plt.barh(final_df["Feature"], final_df["Mean_SHAP"])
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"Top {top_n} SHAP Features (Logistic Regression)")
    plt.gca().invert_yaxis()  # Show the most important feature at the top
    plt.show()


# 7. Filter data by version
def filter_data_by_version(data: pd.DataFrame, major_version: str) -> pd.DataFrame:
    """
    Filtre les données pour inclure uniquement les lignes correspondant à une version majeure spécifique.

    :param data: DataFrame complet contenant toutes les versions.
    :param major_version: Numéro de la version majeure à inclure (ex: "2.0").
    :return: DataFrame filtré.
    """
    return data[data['Version'].str.startswith(major_version)]
