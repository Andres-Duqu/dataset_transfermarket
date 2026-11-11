import argparse
import mlflow.sklearn
import pandas as pd
from src.config import TRACKING_URI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="Ruta al archivo CSV de datos para predecir")
    parser.add_argument("--alias", type=str, default="challenger", help="Alias o versión del modelo a usar")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)

    # Cargar el modelo desde el registro usando el nombre y alias/versión
    model_uri = f"models:/ModeloJugadores@1" if args.alias == "champion" else f"models:/ModeloJugadores/1"
    
    print(f"Cargando modelo desde: {model_uri}")
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Intentar cargar por versión directa si el alias no está configurado
        model_uri = "models:/ModeloJugadores/1"
        model = mlflow.sklearn.load_model(model_uri)

    # Cargar datos de prueba
    df = pd.read_csv(args.data_path)
    
    # Seleccionar las mismas características usadas en el entrenamiento
    X_pred = df[['height_in_cm', 'highest_market_value_in_eur']].dropna()

    predictions = model.predict(X_pred)
    df_result = X_pred.copy()
    df_result['predicted_class'] = predictions

    print("Primeras predicciones:")
    print(df_result.head(10))

if __name__ == "__main__":
    main()
