import pandas as pd
import shutil
import os

# Rutas de los archivos CSV
wildclip_csv = 'Predictions_trans_test_wildclip.csv'   # Cambia la ruta a tu archivo wildclip
bioclip_csv = 'Predictions_trans_test_bioCLip.csv'     # Cambia la ruta a tu archivo bioclip
catalog_csv = 'Predictions_trans_test_CATALOG.csv'     # Cambia la ruta a tu archivo CATALOG

# Directorio base donde están las imágenes
base_image_dir = 'data/terra/img/trans_test/'

# Directorio donde se guardarán las imágenes seleccionadas
output_image_dir = 'CATALOG_Winning_Cases'

# Crear el directorio si no existe
os.makedirs(output_image_dir, exist_ok=True)

# Diccionario de índices de clases a nombres
class_indices = {
    0: 'badger', 1: 'bird', 2: 'bobcat', 3: 'car', 4: 'cat', 5: 'coyote', 6: 'deer', 7: 'dog', 8: 'empty', 9: 'fox',
    10: 'opossum', 11: 'rabbit', 12: 'raccoon', 13: 'rodent', 14: 'skunk', 15: 'squirrel'
}

# Diccionario inverso para mapear nombres a índices
indices_class = {v: k for k, v in class_indices.items()}

# Lee los CSVs con index_col para mantener 'class' y 'Prediction' como filas
wildclip_df = pd.read_csv(wildclip_csv, index_col=0)
bioclip_df = pd.read_csv(bioclip_csv, index_col=0)
catalog_df = pd.read_csv(catalog_csv, index_col=0)

# Identifica las imágenes donde 'class' y 'Prediction' son diferentes en wildclip y bioclip
# pero son iguales en CATALOG
incorrect_wildclip = (wildclip_df.loc['class'] != wildclip_df.loc['Prediction'])
incorrect_bioclip = (bioclip_df.loc['class'] != bioclip_df.loc['Prediction'])
correct_catalog = (catalog_df.loc['class'] == catalog_df.loc['Prediction'])

# Filtra las imágenes que cumplen todas las condiciones
filtered_images = wildclip_df.columns[incorrect_wildclip & incorrect_bioclip & correct_catalog]

# Crear una lista para almacenar los resultados
results = []

# Iterar sobre las imágenes filtradas para construir la tabla
for image in filtered_images:
    expected = catalog_df.loc['class', image]
    catalog_predict = catalog_df.loc['Prediction', image]
    wildclip_predict = wildclip_df.loc['Prediction', image]
    bioclip_predict = bioclip_df.loc['Prediction', image]

    # Reemplazar índices con nombres de clases usando class_indices
    expected_name = class_indices.get(expected, "Unknown")
    catalog_predict_name = class_indices.get(catalog_predict, "Unknown")
    wildclip_predict_name = class_indices.get(wildclip_predict, "Unknown")
    bioclip_predict_name = class_indices.get(bioclip_predict, "Unknown")

    # Añadir los resultados a la lista
    results.append([image, expected_name, catalog_predict_name, wildclip_predict_name, bioclip_predict_name])

    # Construir la ruta de la imagen original y la nueva ruta
    original_image_path = os.path.join(base_image_dir, catalog_predict_name, f"{image}.jpg")  # Asume que las imágenes tienen extensión .jpg
    output_image_path = os.path.join(output_image_dir, f"{image}.jpg")

    # Copiar la imagen a la carpeta de resultados si existe
    if os.path.exists(original_image_path):
        shutil.copy(original_image_path, output_image_path)
    else:
        print(f"Imagen no encontrada: {original_image_path}")

# Crear un DataFrame con los resultados
results_df = pd.DataFrame(results, columns=['name_ima', 'Expected', 'Catalog_predict', 'WildClip_predict', 'BioCLIP_predict'])

# Guardar el resultado en un archivo CSV
output_file = 'resultados_comparativos.csv'
results_df.to_csv(output_file, index=False)

print(f"Tabla comparativa guardada en '{output_file}'. Las imágenes seleccionadas se han copiado a '{output_image_dir}'.")