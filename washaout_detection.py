import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# ==========================================
# 1. SIMULACIÓN DE DATOS DEL POZO (WITSML)
# ==========================================
# Simulamos 10 minutos de perforación normal y 2 minutos de Washout
print("Generando datos de perforación en tiempo real...")

# Datos normales: SPP estable (~3000 psi), Flow Out estable (~500 gpm)
normal_data = {
    'Time_min': np.arange(1, 11),
    'SPP_psi': np.random.normal(3000, 20, 10),
    'Flow_Out_gpm': np.random.normal(500, 5, 10)
}
df_normal = pd.DataFrame(normal_data)

# Datos de Washout: SPP cae drásticamente, Flow Out se mantiene
washout_data = {
    'Time_min': np.arange(11, 13),
    'SPP_psi': [2700, 2400], # Caída de presión
    'Flow_Out_gpm': [498, 501] # Flujo de salida constante (No es pérdida de circulación)
}
df_washout = pd.DataFrame(washout_data)

# Unimos los datos en un solo DataFrame (Historial del pozo)
df_well = pd.concat([df_normal, df_washout], ignore_index=True)

# ==========================================
# 2. ENTRENAMIENTO DE LA INTELIGENCIA ARTIFICIAL
# ==========================================
print("Entrenando modelo de Machine Learning (Isolation Forest)...")

# Seleccionamos las variables (Features) que la IA va a analizar
X = df_well[['SPP_psi', 'Flow_Out_gpm']]

# Inicializamos el modelo de Detección de Anomalías
# contamination=0.15 significa que estimamos que el 15% de los datos podrían ser fallas
ai_model = IsolationForest(contamination=0.15, random_state=42)

# La IA aprende el comportamiento "Normal" del pozo
ai_model.fit(X)

# ==========================================
# 3. PREDICCIÓN Y DETECCIÓN DEL WASHOUT
# ==========================================
# Le pedimos a la IA que evalúe todos los datos
# Devolverá 1 si es Normal, y -1 si es una Anomalía (Washout)
df_well['AI_Anomaly_Alert'] = ai_model.predict(X)

print("\n--- REPORTE DE ESTADO DEL POZO ---")
for index, row in df_well.iterrows():
    status = "🟢 NORMAL" if row['AI_Anomaly_Alert'] == 1 else "🔴 ALERTA DE WASHOUT!"
    print(f"Minuto {int(row['Time_min'])} | SPP: {int(row['SPP_psi'])} psi | Flow Out: {int(row['Flow_Out_gpm'])} gpm | Estado: {status}")