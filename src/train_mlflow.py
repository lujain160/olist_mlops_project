import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from src.config import load_config
from src.data import load_data

def train_and_track():
    # Load configuration
    config = load_config()
    random_state = config["model_params"]["random_state"]

    # Correct the models directory path key
    models_dir = Path(config["Paths"]["model_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data for training and MLflow tracking...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # Set the MLflow experiment name
    mlflow.set_experiment("Olist_Shipping_Delay_Experiment")

    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            random_state=random_state,
            max_iter=1000
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        )
    }

    best_score = 0.0
    best_model_name = None
    best_model = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training and evaluating the model: {name}...")

            model.fit(X_train, y_train)

            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = model.predict(X_val)

            roc_auc = roc_auc_score(y_val, y_pred_proba)
            f1 = f1_score(y_val, y_pred)
            acc = accuracy_score(y_val, y_pred)

            # Log parameters
            mlflow.log_param("model_name", name)
            mlflow.log_param("random_state", random_state)
            if name == "RandomForest":
                mlflow.log_param("n_estimators", 100)

            # Log metrics for all models
            mlflow.log_metric("val_roc_auc", roc_auc)
            mlflow.log_metric("val_f1", f1)
            mlflow.log_metric("val_accuracy", acc)

            # Log the model artifact in MLflow
            mlflow.sklearn.log_model(model, "model")

            print(f"Finished {name} | ROC-AUC: {roc_auc:.4f}")

            # Compare and track the best model based on ROC-AUC
            if roc_auc > best_score:
                best_score = roc_auc
                best_model_name = name
                best_model = model

    # Save the best model locally
    if best_model:
        best_model_path = models_dir / "final_model.joblib"
        joblib.dump(best_model, best_model_path)
        print(f"The winning model is [{best_model_name}] with best ROC-AUC: {best_score:.4f}")
        print(f"Saved the best model to: {best_model_path}")

if __name__ == "__main__":
    train_and_track()
