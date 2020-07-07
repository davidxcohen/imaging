import numpy as np
import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR+'/..'+'/func'))

from func.photonic_func import Photonic
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

# Jupyter lab --> not required for Jupyter notebook
# pip install jupyterlab==1.2.0
# jupyter labextension install @jupyterlab/plotly-extension
# jupyter labextension install @jupyterlab/toc

# Plotly versions keep changing fast
# Plotly version 4.0.0 (pip install plotly==4.0.0)
# init_notebook_mode(connected=True)  # for Jupyter Lab notebook

photonic = Photonic(config='Cfg2')

dist_vec = np.array([1, 2, 5, 10, 20, 50, 100])
trace0 = go.Scatter(x=dist_vec,
                    # Select 'lines', 'markers' or 'lines+markers'
                    y=1000 * photonic.wallFlux(dist_vec=dist_vec), mode='lines+markers',
                    name='wallFlux [W/m**2]')
trace1 = go.Scatter(x=dist_vec,
                    y=1000 * photonic.siliconFlux2(dist_vec=dist_vec), mode='lines+markers',
                    name='siliconFlux [W/m**2]')
trace2 = go.Scatter(x=dist_vec,
                    y=photonic.photoelectron2(dist_vec=dist_vec), mode='lines+markers',
                    name='photoelectrons / burst')
trace3 = go.Scatter(x=[3],
                    y=[10000], mode='text', textposition='top right',
                    name='text', text=['Simulation of light created by a light source attached to a camera<br>'
                                       + '1. Flux is calculated on a wall at a certain distance assuming CW lighting mode<br>'
                                       + '2. Flux on the focal plane of the silicon sensor as imaged from the wall thru the lens<br>'
                                       + '3. Photoelectrons per a single burst collect in the photodiode of the pixel'])

data = [trace0, trace1, trace2, trace3]

layout = {'title': 'Photonic simulation - Flux on wall/sensor and PE count',
          'xaxis': {'title': 'Wall Distance [m]',
                    'type': 'log'},  # Select 'log' or 'linear'
          'yaxis': {'title': 'Flux [mW/m**2], PE [e-]',
                    'type': 'log'},  # Select 'log' or 'linear'
          'template': 'plotly_dark',
          'barmode': 'group',
          'hovermode': 'x'}

iplot({'data': data, 'layout': layout})
