import numpy as np

import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(sys.path)
sys.path.append(os.path.dirname(CURRENT_DIR+'/..'+'/func'))
print('\n ===== \n',sys.path)
from func.photonic_func import Photonic
# Jupyter lab --> not required for Jupyter notebook
# pip install jupyterlab==1.2.0
# jupyter labextension install @jupyterlab/plotly-extension
# jupyter labextension install @jupyterlab/toc  

# Plotly versions keep changing fast
# Plotly version 4.0.0 (pip install plotly==4.0.0)
from plotly.offline import iplot
import plotly.graph_objs as go
# init_notebook_mode(connected=True)  

photonic = Photonic()

photonic = Photonic(config='Cfg2')

dist_vec = np.array([1,2,5,10,20,50,100])
trace0 = go.Scatter(x=dist_vec,
                    y=1000 * photonic.wallFlux(dist_vec=dist_vec), mode='lines+markers',  # Select 'lines', 'markers' or 'lines+markers'
                    name='wallFlux')
trace1 = go.Scatter(x=dist_vec,
                    y=1000 * photonic.siliconFlux2(dist_vec=dist_vec), mode='lines+markers',
                    name='siliconFlux')
trace2 = go.Scatter(x=dist_vec,
                    y=photonic.photoelectron2(dist_vec=dist_vec), mode='lines+markers',
                    name='photoelectrons')

data = [trace0, trace1, trace2]

layout = {'title': 'Flux on wall/sensor and PE count',
          'xaxis': {'title': 'dist [m]',
                    'type': 'log'},  # Select 'log' or 'linear'
          'yaxis': {'title': 'Flux [mW/m**2], PE [e-]',
                    'type': 'log'},  # Select 'log' or 'linear'
          'template': 'plotly_dark'}

iplot({'data': data, 'layout': layout})