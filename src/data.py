import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_PATH

def load_data():
    df = pd.read_csv(DATA_PATH)
    # Ejemplo de preprocesamiento para clasificar 'position' o 'market_value_in_eur'
    # Filtremos filas válidas para position y algunas características numéricas
    df = df.dropna(subset=['position', 'market_value_in_eur', 'height_in_cm'])
    
    # Crear una variable objetivo binaria de ejemplo (ej: si vale más de 1 millón de euros o clasificar posición principal)
    # O clasificamos posición: Defender, Midfield, Attack, Goalkeeper
    features = ['height_in_cm', 'market_value_in_eur', 'highest_market_value_in_eur']
    df = df.dropna(subset=features)
    
    X = df[['height_in_cm', 'highest_market_value_in_eur']]
    # Target: market_value_in_eur alto (> 1 millon) o posición
    y = (df['market_value_in_eur'] > 1000000).astype(int)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)
