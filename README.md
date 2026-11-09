# Drilling Anomaly Detection: Washout vs Lost Circulation 🛢️💻

## 📌 The Problem (El Problema)
Non-Productive Time (NPT) costs the Oil & Gas industry millions of dollars per well. When Standpipe Pressure (SPP) drops, surface algorithms often struggle to differentiate between a **Washout** (hole in the drill string) and **Lost Circulation** (formation fracture). 

As a former Senior Directional Driller, I know that misdiagnosing this event leads to wrong decisions on the rig floor.

## 💡 The Solution (La Solución)
This project uses **Python and Machine Learning (Isolation Forest)** to analyze surface data in real-time. By correlating the SPP gradient with the Flow Out variance, the AI can predict and classify the anomaly before a catastrophic failure occurs.

## 🛠️ Technologies Used (Tecnologías Usadas)
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Anomaly Detection)
* **Database:** SQL (for historical data extraction)

## 📊 How it works (Cómo funciona)
1. Extracts real-time drilling data (WOB, RPM, SPP, Flow In, Flow Out).
2. Calculates the rate of change (gradient) of the SPP.
3. The ML model flags an anomaly if SPP drops while Flow Out remains constant (Washout signature).
