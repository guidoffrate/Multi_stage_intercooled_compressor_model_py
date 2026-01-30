# Compressor Series Simulation Package

This project simulates and analyzes the performance of compressor series using Python.

The model was developed for the following publication (please acknowledge our efforts by citing this and/or the publications reported below):

CHALLENGES DUE TO AMBIENT TEMPERATURE VARIATION FOR ALLAM CYCLE COMPRESSION TRAIN, Alessio Ciambellotti, Guido Francesco Frate, Lorenzo Ferrari, 2026, Proceedings of ASME Turbo Expo 2026, Turbomachinery Technical Conference and Exposition, GT2026, June 15-19, 2026, Milan, Italy, DOI: To be added Upon final Publication



At the core, there is a Python implementation of a centrifugal compressor model with a vaned diffuser. The compressor model is 1-D and solves the impeller, vaned diffuser, and volute equations to simulate compressor performance, given the machine's geometric specifications.

The model accounts for real-gas effects in fluid-property calculations and solves the gas-path equations without relying on perfect-gas assumptions. Coolprop and/or REFPROP are used for the fluid property calculations. 

To model a multi-stage machine, the individual stages are connected in series to form an intercooled compressor, with the outlet pressure from upstream stages used as the input to downstream stages. The Multistage compressor model enables optimization of each stage design by specifying different flow coefficients and tip-speed Mach numbers for each stage. Part of the geometry is fixed across the stages, whereas radii and other dimensions scale with the impeller's outlet radius used in each stage. The possibility of directly specifying the stages' radii and rotational speeds, e.g., to simulate several stages on a single shaft, is also implemented but not used in the current version.

Finally, code for post-processing and plotting results is also provided. The main purpose of these is to plot Ts diagrams of the compressor train.

The model represents an extension of the work presented in the following publications, which can also be cited:

Frate, G.F., Benvenuti, M., and Ferrari, L. (2025). Multi-objective optimised preliminary design of centrifugal compressors for Brayton high-temperature heat pumps, Energy Conversion and Management, doi: [10.1016/j.enconman.2025.120881] (https://doi.org/10.1016/j.enconman.2025.120881)

Frate, G.F., Benvenuti, M., Chini, F., and Ferrari, L. (2024).
Optimal design of centrifugal compressors in small-size high-temperature Brayton heat pumps,
Proceedings of 37th International Conference on Efficiency, Cost, Optimization, Simulation and Environmental Impact of Energy Systems (ECOS), Rhodes, Greece, 30 June - 5 July 2024, doi: 10.52202/077185-0031

A Python version of an earlier version of the model based on the perfect gas hypothesis and using vaneless diffuser can be found at [https://github.com/guidoffrate/Centrifugal-compressor-model](https://github.com/guidoffrate/centrifugal_compressor_model_py)

A Matlab version of an earlier version of the model based on the perfect gas hypothesis and using vaneless diffuser can be found at [https://github.com/guidoffrate/Centrifugal-compressor-model](https://github.com/guidoffrate/Centrifugal-compressor-model)


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
