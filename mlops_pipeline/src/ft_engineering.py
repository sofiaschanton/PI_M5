# Librerías
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


# Cargar datos
data_path = Path("../../Base_de_datos.xlsx")

print("CWD:", Path.cwd())
print("Buscando archivo en:", data_path.resolve())

if not data_path.exists():
    raise FileNotFoundError(f"No se encontró el archivo en: {data_path.resolve()}")

df = pd.read_excel(data_path)

print("✅ Dataset cargado")

