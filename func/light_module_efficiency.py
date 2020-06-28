import numpy as np 
from scipy.interpolate import UnivariateSpline

def light_efficiency(electric_current, electric_voltage, radiant_power, light_type):
	

	spl_IV = UnivariateSpline(electric_current, electric_voltage)
	spl_IP = UnivariateSpline(electric_current, radiant_power)

	I = np.linspace(np.amin(electric_current), np.amax(electric_current), num=200)
	V = spl_IV(I)
	P = spl_IP(I) # Optical radiant power

	light_electrical_power = I * V

	wall_plug_efficiency = P / light_electrical_power
	heat_dissipation = light_electrical_power - P
	return I, V, P, wall_plug_efficiency, heat_dissipation


if __name__ == '__main__':
	# Example source data from:
	# https://www.hamamatsu.com/resources/pdf/ssd/l12509_series_kled1071e.pdf
	# P/N = L12509-0155K 
	light_type = 'Hamamatsu 1550nm L12509-0155K'
	I = np.array([10,  20,   50,   100, 200,  500]) * 1e-3 # [A]
	V = np.array([0.7, 0.73, 0.78, 0.9, 1.07, 1.4]) # [V]
	P = np.array([0.5, 0.9,  1.9,  3.2, 5.5,  8.2]) * 1e-3 # [W]

	I, V, P, efficiency, heat = light_efficiency(electric_current=I, electric_voltage=V, radiant_power=P, light_type=light_type)

	import matplotlib.pyplot as plt
	plt.style.use('dark_background')
	fig, ax = plt.subplots(2, 2, figsize=(12,6))
	

	ax[0, 0].plot(I, V, label='I-V:'+light_type)
	ax[0, 0].set_xlabel('Forward current [A]')
	ax[0, 0].set_ylabel('Voltage [V]')
	ax[0, 0].grid(True, color='blue')
	ax[0, 0].legend(fontsize=8)

	ax[1, 0].plot(I, P, label='I-P:'+light_type)
	ax[1, 0].set_xlabel('Forward current [A]')
	ax[1, 0].set_ylabel('Optical Radiant Power [W]')
	ax[1, 0].grid(True, color='blue')
	ax[1, 0].legend(fontsize=8)

	ax[0, 1].plot(I, efficiency, label='Wall Plug Efficiency:'+light_type)
	ax[0, 1].set_xlabel('Forward current [A]')
	ax[0, 1].set_ylabel('Efficency [ratio]')
	ax[0, 1].grid(True, color='blue')
	ax[0, 1].legend(fontsize=8)

	ax[1, 1].plot(I, heat, label='Heat Disspation:'+light_type)
	ax[1, 1].set_xlabel('Forward current [A]')
	ax[1, 1].set_ylabel('Heat dissipation [W]')
	ax[1, 1].grid(True, color='blue')
	ax[1, 1].legend(fontsize=8)	
	
	# plt.show()

	from plotly.offline import iplot
	import plotly.graph_objs as go
	# Plotly version 4.0.0 (pip install plotly==4.0.0)

	trace0 = go.Scatter(x=I,
	                    y=V, mode='lines+markers',  # Select 'lines', 'markers' or 'lines+markers'
	                    name='Forward Voltage [V]')
	trace1 = go.Scatter(x=I,
	                    y=1000 * P, mode='lines+markers',
	                    name='Optical Radiant Power [mW]')
	trace2 = go.Scatter(x=I,
	                    y=100 * efficiency, mode='lines+markers',
	                    name='Wall Plug Efficiency [%]')
	trace3 = go.Scatter(x=I,
	                    y=1000 * heat, mode='lines+markers',
	                    name='Heat Dissipatiom [mW]')
	data = [trace0, trace1, trace2, trace3]

	layout = {'title': 'Light Component Efficiency Analysis: '+light_type,
	          'xaxis': {'title': 'Forward Current [A]',
	                    'type': 'linear'},  # Select 'log' or 'linear'
	          'yaxis': {'title': '____',
	                    'type': 'log'},  # Select 'log' or 'linear'
	          'template': 'plotly_dark',
	          'hovermode': 'x'}

	iplot({'data': data, 'layout': layout})

