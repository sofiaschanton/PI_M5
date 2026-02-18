import os
import pandas as pd

def cargar_datos(verbose: bool = False):

    # 1. Ruta absoluta del directorio donde está este script.
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    if verbose:
        print("Ruta actual:", ruta_actual)

    # 2. Ruta absoluta del proyecto (subir 2 niveles).
    ruta_proyecto = os.path.dirname(ruta_actual)
    if verbose:
        print("Ruta proyecto (1 nivel arriba):", ruta_proyecto)
    
    # 3. Construir la ruta completa al archivo Excel
    ruta_excel = os.path.join(ruta_proyecto, "Base_de_datos.xlsx")
    if verbose:
        print("Ruta Excel:", ruta_excel)

    # 4. Leer los datos
    df = pd.read_excel(ruta_excel)
    if verbose:
        print(df.head())
    return df


if __name__ == "__main__":
    datos = cargar_datos(verbose=True)
    print(datos.head())
    print(datos.columns)