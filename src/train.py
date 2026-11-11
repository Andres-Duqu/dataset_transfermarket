import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.data import load_data
from src.config import TRACKING_URI, EXPERIMENT_NAME, MODEL_NAME

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-estimators", type=int, default=50, help="Number of estimators")
    parser.add_argument("--max-depth", type=int, default=3, help="Max depth")
    parser.add_argument("--run-name", type=str, default="probando", help="Run name")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X_train, X_test, y_train, y_test = load_data()

    with mlflow.start_run(run_name=args.run_name):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)

        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Registrar automáticamente con el nombre correcto MODEL_NAME ("jugadores_modelo")
        mlflow.sklearn.log_model(
            model, 
            name="model", 
            skops_trusted_types=["sklearn.tree._tree.Tree"],
            registered_model_name=MODEL_NAME
        )
        print(f"Run completed! Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
