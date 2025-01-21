import wandb
import pandas as pd

# Autenticarse con wandb
wandb.login(key="282780c770de0083eddfa3c56402f555ee60e108")

# Configurar proyecto y sweep
project_name = "Ablation_Tem_vs_Des"
sweep_id = "5h49ix6s"  # Reemplaza con el ID de tu sweep (lo encuentras en la URL de la sweep)

# Obtener la API de wandb
api = wandb.Api()

# Cargar los resultados de la sweep
sweep = api.sweep(f"{project_name}/{sweep_id}")
runs = sweep.runs

# Crear un DataFrame con los resultados
data = []
for run in runs:
    row = run.summary._json_dict  # Resultados de la ejecución
    row.update(run.config)       # Configuración usada
    row["name"] = run.name       # Nombre de la ejecución
    data.append(row)

df = pd.DataFrame(data)

# Guardar los resultados como CSV
df.to_csv("1.0_results.csv", index=False)

print("Resultados exportados a sweep_results.csv")
