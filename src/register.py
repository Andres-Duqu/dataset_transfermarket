import mlflow
from mlflow.tracking import MlflowClient
from src.config import TRACKING_URI, EXPERIMENT_NAME, MODEL_NAME

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        print(f"No se encontró el experimento {EXPERIMENT_NAME}")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )

    if not runs:
        print("No se encontraron ejecuciones.")
        return

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    
    result = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    print(f"Modelo registrado exitosamente: {MODEL_NAME}, versión: {result.version}")

    try:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="challenger",
            version=result.version
        )
        print(f"Alias 'challenger' asignado a la versión {result.version}")
    except Exception as e:
        print(f"Error al asignar alias: {e}")

if __name__ == "__main__":
    main()
