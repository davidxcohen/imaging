import numpy as np
import os, sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR+'/..'+'/func'))

from func.photonic_func import Photonic

# Plotly versions keep changing fast
# Plotly version 4.0.0 (pip install plotly==4.0.0)
from plotly.offline import init_notebook_mode, iplot 
import plotly.graph_objs as go
# init_notebook_mode(connected=True)  # for Jupyter Lab notebook

photonic = Photonic(config='Cfg3')

# Light & Shutter pulses + its Convolution: both square and equal
rise = 1e-14
fall = 1e-14
width = 1e-8
y1, t1 = photonic.generate_pulse(rise=rise, fall=fall, width=width, smooth=False)

rise = 1e-14
fall = 1e-14
width = 1e-8
delay = 3e-9
y2, t2 = photonic.generate_pulse(delay=delay, rise=rise, fall=fall, width=width, smooth=False)

y3, t3 = photonic.conv_light_shutter(t_light=t1, y_light=y1, t_shutter=t2, y_shutter=y2)

trace0 = go.Scatter(x=t1,
                    y=y1, mode='lines+markers',  # Select 'lines', 'markers' or 'lines+markers'
                    name='Light')
trace1 = go.Scatter(x=t2,
                    y=y2, mode='lines+markers',
                    name='Shutter')
trace2 = go.Scatter(x=t3-width-2*1e-9,
                    y=y3, mode='lines+markers',
                    name='Conv light-shutter')
trace3 = go.Scatter(x=[-13e-9],
                    y=[0.8], mode='text', textposition='top right',
                    name='text', text=['Convolution is normalized to the light integral' 
                                       + '<br>Equal 1.0 when light fully integrated by the shutter'])

data = [trace0, trace1, trace2, trace3]

layout = {'title': 'Light & Shutter pulses + its Convolution: both square and equal',
          'xaxis': {'title': 'time, time delay [sec]',
                    'type': 'linear'},  # Select 'log' or 'linear'
          'yaxis': {'title': 'Signal',
                    'type': 'linear'},  # Select 'log' or 'linear'
          'template': 'plotly_dark'}

iplot({'data': data, 'layout': layout})

# Light & Shutter pulses + its Convolution: square shutter triangle light
rise = 1e-8
fall = 1e-8
width = 1e-8
y1, t1 = photonic.generate_pulse(rise=rise, fall=fall, width=width, smooth=False)

rise = 1e-14
fall = 1e-14
width = 0.8e-8
delay = 3e-9
y2, t2 = photonic.generate_pulse(delay=delay, rise=rise, fall=fall, width=width, smooth=False)

y3, t3 = photonic.conv_light_shutter(t_light=t1, y_light=y1, t_shutter=t2, y_shutter=y2)

trace0 = go.Scatter(x=t1,
                    y=y1, mode='lines+markers',  # Select 'lines', 'markers' or 'lines+markers'
                    name='Light')
trace1 = go.Scatter(x=t2,
                    y=y2, mode='lines+markers',
                    name='Shutter')
trace2 = go.Scatter(x=t3-width-2*1e-9,
                    y=y3, mode='lines+markers',
                    name='Conv light-shutter')
trace3 = go.Scatter(x=[-10e-9],
                    y=[0.8], mode='text', textposition='top right',
                    name='text', text=['Convolution is normalized to the light integral' 
                                       + '<br>Equal 1.0 when light fully integrated by the shutter'])

data = [trace0, trace1, trace2, trace3]

layout = {'title': 'Light & Shutter pulses + its Convolution: square shutter triangle light',
          'xaxis': {'title': 'time, time delay [sec]',
                    'type': 'linear'},  # Select 'log' or 'linear'
          'yaxis': {'title': 'Signal',
                    'type': 'linear'},  # Select 'log' or 'linear'
          'template': 'plotly_dark'}

iplot({'data': data, 'layout': layout})