import os
import pandas as pd

def cargar_datos():
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    ruta_excel = os.path.join(
        ruta_actual,
        '..', '..',  # 👈 subir 2 niveles
        'Base_de_datos.xlsx'
    )

    print(ruta_excel)                # debug
    print(os.path.exists(ruta_excel))# debe ser True

    df = pd.read_excel(ruta_excel)
    return df

if __name__ == "__main__":
    datos = cargar_datos()
    print(datos.head())