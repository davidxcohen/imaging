import numpy as np
import pandas as pd
import sys
from scipy import signal

nm = 1e-9
c = 2.99792458e8  # [m/s] Speed of light
hc = 1.987820871E-025  # [J * m / photon] Energy of photon with wavelength m
efficacy = 683 # [lumen/watt @ 550nm]
k_b = 1.3806488e-23 # [J/K] boltzman constant
h   = 6.62606957e-34 # [J*sec] Plank constant
nm = 1e-9 # [m]
cm = 1e-2 # [m]
e_minus =1.60217662e-19  # [Culomb]


def PlankLawBlackBodyRad(T, wavelength):
  # Calculates black body radiation according Plank Law. T is the black body
  # temperatue in Kelvin.  Wavelength in Meter. B is the
  # Radiance in [W/sr/m^3]. W is the Spectral radiance in [W/sr/m^2/nm],
  # meaning, the power emitted per unit area of the body, per unit solid
  # angle that the radiation is measured over, per nm of wavelength.
  # 
  # By David Cohen 2009
  # modified to function by DC 14-Jan-2015
  # python - DC 28-Aug-2019

  # Usage:
  # nm      = 1e-9;
  # wavelength  = 850*nm;
  # T = 3200 [K]
  # W       = PlankLawBlackBodyRad(T, wavelength)

  B   = 2 * h * c**2 /  wavelength**5 /(np.exp(hc/(wavelength * k_b * T)) - 1) # [W/sr/m^3] Radiance
  # http://en.wikipedia.org/wiki/Planck%27s_law
  return B * 1e-9 # [W/sr/m^2/nm] Spectral Radiance

class Photonic:
	def __init__(self, config=None):
		self.config = config
		if config is None:
			self.config = 'Cfg1'

		# Read excel table having the following sheets: 'Light', 'Sensor', 'Scene', 'Lens', 'Op', 'Config'
		data_file = '../data/photonic_simul_data.xlsx'
		self.light_ = pd.read_excel(data_file,sheet_name='Light',header=1,index_col='Name')
		self.sensor_ = pd.read_excel(data_file,sheet_name='Sensor',header=1,index_col='Name')
		self.scene_ = pd.read_excel(data_file,sheet_name='Scene',header=1,index_col='Name')
		self.lens_ = pd.read_excel(data_file,sheet_name='Lens',header=1,index_col='Name')
		self.op_ = pd.read_excel(data_file,sheet_name='Op',header=1,index_col='Name')
		self.config_ = pd.read_excel(data_file,sheet_name='Config',header=1,index_col='Name')

		# check Excel data validity
		try:			
			# Reduce parameters level to those appear in the config, only.
			# print(' ## Photonic ## \n', config_.loc[self.cfg],'\n =====  \n' )
			self.light = self.light_.loc[self.config_.loc[self.config , 'Light']]
			self.scene = self.scene_.loc[self.config_.loc[self.config , 'Scene']]
			self.lens = self.lens_.loc[self.config_.loc[self.config , 'Lens']]
			self.sensor = self.sensor_.loc[self.config_.loc[self.config , 'Sensor']]
			self.op = self.op_.loc[self.config_.loc[self.config , 'Op']]
		except KeyError as err:
			print('\033[91m'+'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
			print('Photonic::KeyError::Configutation \033[106m{}\033[0m\033[91m is mismatch key:\033[106m{}'.format(self.config, err)+ '\033[0m')
			sys.exit(1)

		self.wall_flux = self.wallFlux()
		self.silicon_flux = self.siliconFlux(self.wall_flux)

	def wallFlux(self, light=None, scene=None, dist_vec=None, light_type='ir_pulse'):
		if light is None:
			light = self.light
		if scene is None:
			scene = self.scene	
		if dist_vec is None:
			dist_m = scene.Distance_m
		else:
			dist_m = dist_vec

		# print(' ## wallFlux ## \n', light.PeakPower_W,'\n =====  \n' )
		# Flux on a wall (scene) during lighting time (pulse time)
		# Flux due to point source attached to the sensor
		return light.PeakPower_W * light.Transmission / (np.radians(light.Hfov_deg) * np.radians(light.Vfov_deg) * dist_m ** 2)  # [W/m^2]

	def siliconFlux(self, wall_flux, lens=None, scene=None):
		if lens is None:
			lens = self.lens
		if scene is None:
			scene = self.scene			
		# Flux on silicon from the scene during lighting time (pulse time)
		# Flux due to point source attached to the sensor
		# flux_active = light['peak'] * light['transmission'] / (light['hfov'] * light['vfov'] * scene['distance'] ** 2)  # [W/m^2]
		return wall_flux * scene.Reflectivity * lens.Transmission / (4 * lens.F_num ** 2)   # [W/m^2]

	def photoelectron(self, siliconFlux, sensor=None, light=None, op=None):
		if sensor is None:
			sensor = self.sensor
		if light is None:
			light = self.light
		if op is None:
			op = self.op
			
		energy_to_pe = hc / (light.WaveLength_nm * nm) # Conversion from energy [J] to number of photons
		# PE on silicon from the scene during lighting time (pulse time)
		pe_per_sec = siliconFlux * (sensor.PixelSize_m ** 2) * sensor.QE * sensor.FF / energy_to_pe  # [photoelectrons / sec] 
		pe_per_burst = pe_per_sec * op.InBurstDutyCycle * op.BurstTime_s  # [photoelectrons in burst]
		return pe_per_burst

	def siliconFlux2(self, light=None, scene=None, lens=None, dist_vec=None):
		return self.siliconFlux(wall_flux=self.wallFlux(dist_vec=dist_vec))

	def photoelectron2(self, light=None, scene=None, lens=None, sensor=None, op=None, dist_vec=None):
		return self.photoelectron(siliconFlux=self.siliconFlux(wall_flux=self.wallFlux(dist_vec=dist_vec)))

	def generate_pulse(self, delay=None, rise=None, fall=None, width=None, time_interval=None, smooth=False, mode='charge_discharge'):
		'''Create a light pulse (similar to a capacitor charge/discharge)
			rise - Pulse rise time [sec]
			fall - Pulse fall time [sec]
		    width - Pulse width time [sec]
		    time_interval - Sample time_interval [sec]
		'''
		self.time_interval = time_interval
		if time_interval is None:
			self.time_interval = 0.1e-9 # [sec]
		if delay is None:
			delay = 0.

		zeros_len = np.uint(10. + delay / self.time_interval)

		if mode == 'charge_discharge':
			x_rise = np.arange(0, width, self.time_interval)
			x_fall = np.arange(0, fall * 10, fall) # Discharge takes longer to become zero
			y_up = (1 - np.exp(-x_rise / rise))
			y_max = y_up.max()
			y_inf = (1 - np.exp(-(10. *rise) / rise)) # =1 @ infinity
			y_up = y_up / y_up.max()
			y_down = np.exp(-x_fall / fall)
			y_down = y_down / y_down.max()
			y_zeros = np.zeros(zeros_len)
		y = np.concatenate((y_zeros, y_up, y_down))

		# Smooth curves
		if smooth:
			f = signal.hamming(15)
			y = np.convolve(f, y, mode='same')
			y = y / y.max()
		y = y * y_max / y_inf

		# Clip negative 
		y[y < 0] = 0

		# Create time vector
		t = np.linspace(0, len(y) * self.time_interval, len(y))
		self.pulse_y = y
		self.pulse_t = t
		return y, t


	def conv_light_shutter(self, t_light=None, y_light=None, t_shutter=None, y_shutter=None, time_interval=None):
		self.time_interval = time_interval
		if time_interval is None:
			self.time_interval = 0.1e-9 # [sec]

		y = np.convolve(y_light, y_shutter, mode='full')
		y = y / y_light.sum() # Normalize convolution to the integrated light: y=1 if the entire illumination pulse is within the shutter
		t = np.linspace(0, len(y) * self.time_interval, len(y))
		return y, t


	def qe_by_responsivity(self, silicon_responsivity_a_w, pixel_fill_factor, wavelength_m):
		''' 
		Silicon responsivity [A/W]
		fill factor [ratio]
		wavelength [m]
		[A]/[W]*[J * m / photon]/[m]/[C] ==> [1]
		'''
		return pixel_fill_factor * silicon_responsivity_a_w * hc / (wavelength_m * e_minus) # [ratio] ff*QE  


class Spectra:
	def __init__():
		a = 1


if __name__ == '__main__':
	config = pd.read_excel('../data/photonic_simul_data.xlsx',sheet_name='Config',header=1,index_col='Name').loc['Cfg1']
	print(' ## main ## \n', config, '\n =====  \n')


	# create photonic variable, run __init__ function
	photonic = Photonic()

	# Print simulation data
	for index, val in photonic.light.iteritems():
		print('light.',index, val)
	print('---')
	for index, val in photonic.scene.iteritems():
		print('scene.',index, val)
	print('---')
	for index, val in photonic.lens.iteritems():
		print('lens.',index, val)
	print('---')
	for index, val in photonic.sensor.iteritems():
		print('sensor.',index, val)
	print('---')
	for index, val in photonic.op.iteritems():
		print('op.',index, val)

	print('------\nwallFlux @ dist_vec    '+' '.join('{}: {:6.3f} '.format(*k) for k in enumerate(photonic.wallFlux(dist_vec=np.array([1, 2, 3])))), 'W/m**2')
	print('siliconFlux @ dist_vec '+' '.join('{}: {:6.4f} '.format(*k) for k in enumerate(photonic.siliconFlux2(dist_vec=np.array([1, 2, 3])))), 'W/m**2')
	print('PE @ dist_vec          '+' '.join('{}: {:6.1f} '.format(*k) for k in enumerate(photonic.photoelectron2(dist_vec=np.array([1, 2, 3])))), 'e-')

	# photonic = Photonic(config='Cfg2')	
	# call function wallFlux
	print('=====\nWall flux = {:2.3} W/m**2\nSilicon flux = {:2.3}  W/m**2\nPhotoelectron = {:5.5} photonelectron/burst\n======'.format(
		photonic.wallFlux(), 
	    photonic.siliconFlux(wall_flux=photonic.wallFlux()),
	    photonic.photoelectron(siliconFlux=photonic.siliconFlux(wall_flux=photonic.wallFlux())))) #,'\n',photonic.photoelectron())

	photonic = Photonic(config='Cfg3')

	tmp = photonic.qe_by_responsivity(silicon_responsivity_a_w=0.002, pixel_fill_factor=1., wavelength_m=1550.*nm)
	print('####%7.7f'%tmp+'#####')

	print('=====\nWall flux = {:2.3} W/m**2\nSilicon flux = {:2.3}  W/m**2\nPhotoelectron = {:5.5} photonelectron/burst\n======'.format(
		photonic.wallFlux(), 
	    photonic.siliconFlux(wall_flux=photonic.wallFlux()),
	    photonic.photoelectron(siliconFlux=photonic.siliconFlux(wall_flux=photonic.wallFlux())))) #,'\n',photonic.photoelectron())	

	rise = photonic.op_.loc[photonic.config_.loc[photonic.config,'Op'],'light_rise_sec']
	fall = photonic.op_.loc[photonic.config_.loc[photonic.config,'Op'],'light_fall_sec']
	width = photonic.op_.loc[photonic.config_.loc[photonic.config,'Op'],'light_width_sec']
	y1, t1 = photonic.generate_pulse(rise=rise, fall=fall, width=width, smooth=True)


	rise = photonic.op_.loc[photonic.config_.loc[photonic.config,'Op'],'shutter_rise_sec']
	fall = photonic.op_.loc[photonic.config_.loc[photonic.config,'Op'],'shutter_fall_sec']
	width = photonic.op_.loc[photonic.config_.loc[photonic.config,'Op'],'shutter_width_sec']
	delay = photonic.op_.loc[photonic.config_.loc[photonic.config,'Op'],'shutter_delay_sec']
	delay = 14e-9
	y2, t2 = photonic.generate_pulse(delay=delay, rise=rise, fall=fall, width=width, smooth=True)

	y3, t3 = photonic.conv_light_shutter(t_light=t1, y_light=y1, t_shutter=t2, y_shutter=y2)

	import matplotlib.pyplot as plt
	plt.plot(t1,y1)
	plt.plot(t2,y2)
	plt.plot(t3, y3)
	plt.show()

	
	

