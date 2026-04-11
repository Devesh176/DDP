import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from src._source import PIELM_MixedBC
import logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
ansys_data = pd.read_csv('data/steady_state/case3.xls', sep='\t', encoding='ISO-8859-1')
ansys_data.columns = ['node', 'x', 'y', 'z', 'temp']

tol = 1e-10
ansys_data['x'] = ansys_data['x'].where(ansys_data['x'].abs() > tol, 0.0)
ansys_data['y'] = ansys_data['y'].where(ansys_data['y'].abs() > tol, 0.0)

x_max = ansys_data['x'].max()
y_max = ansys_data['y'].max()

logger.info(" ANSYS data info")
logger.info(f"x range: {ansys_data['x'].min()} -> {x_max} m")
logger.info(f"y range: {ansys_data['y'].min()} -> {y_max} m")
logger.info(f"T range: {ansys_data['temp'].min()} -> {ansys_data['temp'].max()} C")
logger.info(f"Nodes: {len(ansys_data)}\n")

model = PIELM_MixedBC(hidden_nodes=1800)
model.train(nx=40, ny=40, x_max=x_max, y_max=y_max, T_inf=22., k=50., h=25., q_flux=500.)

X_pred = ansys_data[['x', 'y']].values
T_pred = model.predict(X_pred)

ansys_data['T_pielm'] = T_pred
ansys_data['error']   = ansys_data['temp'] - T_pred

T_scale = ansys_data['temp'].max() - ansys_data['temp'].min()
logger.info(f"Using T_scale = {T_scale} for normalised error")
ansys_data['error_norm'] = ansys_data['error'] / T_scale

logger.info("\nError summary")
logger.info(f" Max |error|: {ansys_data['error'].abs().max()} C")
logger.info(f" Mean |error|: {ansys_data['error'].abs().mean()} C")
logger.info(f" RMS error: {np.sqrt((ansys_data['error']**2).mean())} C")
logger.info(f" Max normalised err: {ansys_data['error_norm'].abs().max()}")
logger.info(f" RMS normalised err: {np.sqrt((ansys_data['error_norm']**2).mean())}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

x = ansys_data['x'].values
y = ansys_data['y'].values
triang = tri.Triangulation(x, y)

def triplot(ax, vals, title, cmap='hot', sym=False):
    vmin = vals.min() if not sym else -np.abs(vals).max()
    vmax = vals.max() if not sym else  np.abs(vals).max()
    tc = ax.tricontourf(triang, vals, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(tc, ax=ax, shrink=0.8)
    ax.set_title(title)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')

triplot(axes[0], ansys_data['temp'].values, 'ANSYS temperature (C)')
triplot(axes[1], ansys_data['T_pielm'].values, 'PIELM temperature (C)')
triplot(axes[2], ansys_data['error'].values, 'Error = ANSYS − PIELM (C)', cmap='RdBu', sym=True)

plt.tight_layout()
plt.savefig('plots/pielm_comparison_steady_case3.png', dpi=150, bbox_inches='tight')
plt.show()
logger.info("\nSaved plots/pielm_comparison_steady_case3.png")