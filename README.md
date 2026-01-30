# Compressor Series Simulation Package

This project simulates and analyses compressor series performance using Python.

---

## 1️⃣ Requirements
- Python 3.12 (Anaconda or Miniconda)
- (Optional but recommended) REFPROP, if you want high-accuracy property data.

---

## 2️⃣ Setup (Anaconda user)

1. **Unzip** the project folder anywhere you like.

2. **Open Anaconda Prompt** and go to that folder:

   cd path\to\project
   

3. **Create and activate an environment:**
   
   conda create -n compressor python=3.12
   conda activate compressor
   

4. **Install dependencies:**
   
   pip install -r requirements.txt
   

---

## 3️⃣ Running the simulations

Example:

python compressor_series_offdesign_massflow_optimised.py


This will run the main off-design analysis and create output tables and plots.

---

## 4️⃣ Notes

- **REFPROP backend:**  
  Some scripts set `backend="REFPROP"`.  
  
  If REFPROP is not available, edit the driver script to use:
 
  backend = "HEOS"  # or "auto"
 
  Everything else will run unchanged.

- **Output folders:**  
  Before running, create check if there is an empty folder named compressor_series_pictures. if there is not create it typing in the therminal:
  
  mkdir compressor_series_pictures
  
  Results (plots, CSVs, etc.) are saved there.

---

## 5️⃣ Project structure

| File | Purpose |
|------|----------|
| `compressor_series_offdesign_massflow_optimised.py` | Main driver script for off-design mass-flow optimisation |
| `simulate_compressor_series.py` | Core compressor series simulator |
| `compressor_model_vaned.py` | Compressor model with vaned geometry |
| `plot_ts_dome_compressor_series.py` | Generates T-s dome plots |
| `plot_compressor_series.py` | Creates summary bar charts |
| `dvdt_contour.py` | Calculates and plots dv/dT contours |
| `compressor_series_pictures/` | Folder where results (plots, CSVs) are saved |

---

## 6️⃣ Quick test

After installing dependencies, test everything works:

python - << "PY"
from simulate_compressor_series import simulate_compressor_series
res = simulate_compressor_series(mdot=300, n_stages=4,
                                 p01_0=30e5, T01_list=[300]*4,
                                 fluid="CO2", backend="HEOS",
                                 show_figure=False)
print("✅ Simulation OK — overall PR:", res["overall"]["PR_tt_series"])
PY


If you see a “Simulation OK” message, the setup is complete.
