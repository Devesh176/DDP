import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from src._source import PIELM_MixedBC_Transient
import logging 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_dir = Path('data/transient_state/case3')
df = []
for file in data_dir.glob('*.xls'):
    logger.info(f"Found ANSYS data file: {file}")

    ansys_data = pd.read_csv(file, sep='\t', encoding='ISO-8859-1')
    ansys_data.columns = ['node', 'x', 'y', 't', 'temp']
    ansys_data = ansys_data[['x', 'y', 't', 'temp']]  
    tol = 1e-10
    ansys_data['x'] = ansys_data['x'].where(ansys_data['x'].abs() > tol, 0.0)
    ansys_data['y'] = ansys_data['y'].where(ansys_data['y'].abs() > tol, 0.0)
    ansys_data['t'] = file.stem.split('_')[-1]  # extract time from filename
    ansys_data['x'] = ansys_data['x'].astype(float)
    ansys_data['y'] = ansys_data['y'].astype(float)
    ansys_data['t'] = ansys_data['t'].astype(float)
    ansys_data['temp'] = ansys_data['temp'].astype(float)
    df.append(ansys_data)

df = pd.concat(df, ignore_index=True)

x_max = df['x'].max()
y_max = df['y'].max()
t_max = df['t'].max()

logger.info(" ANSYS data info")
logger.info(f"x range: {ansys_data['x'].min()} -> {x_max} m")
logger.info(f"y range: {ansys_data['y'].min()} -> {y_max} m")
logger.info(f"t range: {ansys_data['t'].min()} -> {t_max} s")
logger.info(f"T range: {ansys_data['temp'].min()} -> {ansys_data['temp'].max()} C")
logger.info(f"Nodes: {len(ansys_data)}\n")

model = PIELM_MixedBC_Transient(hidden_nodes=5390)
model.train(nx=20, ny=20, x_max=x_max, y_max=y_max, t_max=t_max, k=50., h=25., q_flux=500., T_inf=22.,
              rho=7850., cp=434.,T_initial=22., seed=42)

X_pred = df[['x', 'y', 't']].values
# divide X_pred into batches
per_batch = 10000
T_pred = np.zeros(len(X_pred))
for i in range(0, len(X_pred), per_batch):
    batch_X = X_pred[i:i+per_batch]
    T_pred[i:i+per_batch] = model.predict(batch_X)

df['T_pielm'] = T_pred
df['error'] = df['temp'] - T_pred

# Normalised error 
T_scale = df['temp'].max() - df['temp'].min()
df['error_norm'] = df['error'] / T_scale

for t in sorted(df['t'].unique()):
    df_t = df[df['t'] == t]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    x = df_t['x'].values
    y = df_t['y'].values
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

    triplot(axes[0], df_t['temp'].values,'ANSYS temperature (C)')
    triplot(axes[1], df_t['T_pielm'].values,'PIELM temperature (C)')
    triplot(axes[2], df_t['error'].values,'Error = ANSYS − PIELM (C)', cmap='RdBu', sym=True)

    logger.info(f"\nError summary at {t}")
    logger.info(f"Max |error|: {df_t['error'].abs().max()} C")
    logger.info(f"Mean |error|: {df_t['error'].abs().mean()} C")
    logger.info(f"RMS error: {np.sqrt((df_t['error']**2).mean())} C")
    logger.info(f"Max normalised err: {df_t['error_norm'].abs().max()}") 
    logger.info(f"RMS normalised err: {np.sqrt((df_t['error_norm']**2).mean())}")

    plt.tight_layout()
    plt.savefig(f'plots/pielm_comparison_transient_case3_{t}.png', dpi=150, bbox_inches='tight')
    plt.show()
    logger.info(f"\nSaved plots/pielm_comparison_transient_case3_{t}.png")