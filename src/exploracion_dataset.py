import pandas as pd

def cargar_dataset(ruta):
    df = pd.read_csv(ruta)
    return df

def explorar_dataset(df):

    print("=== PRIMEROS REGISTROS ===")
    print(df.head())

    print("\n=== INFORMACIÓN GENERAL ===")
    print(df.info())

    print("\n=== COLUMNAS ===")
    print(df.columns)

    print("\n=== DIMENSIONES ===")
    print(df.shape)

    print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
    print(df.describe())

    print("\n=== VALORES NULOS ===")
    print(df.isnull().sum())