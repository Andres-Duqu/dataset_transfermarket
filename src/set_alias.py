import mlflow
from mlflow.tracking import MlflowClient
from src.config import TRACKING_URI, MODEL_NAME

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Buscar la última versión del modelo
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        print("No se encontraron versiones del modelo.")
        return

    latest_version = max([int(v.version) for v in versions])
    print(f"Asignando alias 'champion' a la versión {latest_version} de {MODEL_NAME}...")

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=str(latest_version)
    )
    print("¡Alias 'champion' asignado exitosamente!")

if __name__ == "__main__":
    main()
