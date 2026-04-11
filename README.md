# Physics-Informed Extreme Learning Machine (PI-ELM) for Diffusion Equations

This project implements **Physics-Informed Extreme Learning Machines (PI-ELM)** to solve **2D diffusion equations**, along with comparisons against simulated results.

---

## Physical Constants

The following physical parameters are used in the simulations:

- **Thermal Conductivity (k)** = 50 W/m·°C  
- **Internal Heat Generation (q)** = 1 × 10⁹ W/m³  
- **Density (ρ)** = 7850 kg/m³  
- **Specific Heat Capacity (Cp)** = 434 J/kg·°C  
- **Convective Heat Transfer Coefficient (h)** = 25 W/m²·°C  

---

## Project Structure

    .
    ├── data/
    │   ├── steady_state/                        # Ansys Sim data for steady state
    │   └── transient_state/                     # Ansys Sim data for transient state
    ├── plots/                                   # Temperature and error plots
    ├── src/
    │   └── _source.py                           # Source file with all classes
    ├── tests/                                   # All tests
    │   ├── test_2d_diffusion_steady_case1.py
    │   ├── test_2d_diffusion_steady_case2.py
    │   ├── test_2d_diffusion_steady_case3.py
    │   ├── test_2d_diffusion_transient_case1.py
    │   ├── test_2d_diffusion_transient_case2.py
    │   └── test_2d_diffusion_transient_case3.py
    ├── requirements.txt
    └── README.md

