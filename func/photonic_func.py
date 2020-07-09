import numpy as np
import pandas as pd
import sys, os
from scipy import signal, interpolate

nm = 1e-9
c = 2.99792458e8  # [m/s] Speed of light
hc = 1.987820871E-025  # [J * m / photon] Energy of photon with wavelength m
efficacy = 683 # [lumen/watt @ 550nm]
k_b = 1.3806488e-23 # [J/K] boltzman constant
h   = 6.62606957e-34 # [J*sec] Plank constant
nm = 1e-9 # [m]
cm = 1e-2 # [m]
e_minus =1.60217662e-19  # [Culomb]
T_kelvin = 300
maximum_amb_light_lux = 120000 # [lux]


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
		CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
		self.config = config
		if config is None:
			self.config = 'Cfg1'

		# Read excel table having the following sheets: 'Light', 'Sensor', 'Scene', 'Lens', 'Op', 'Config'
		data_file = CURRENT_DIR+'/../data/photonic_simul_data.xlsx'
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

	def solarSpectrum_W_m2_um(self):
		# NASA technical Memorandum1980
		# https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19810016493.pdf
		solar_spectrum_radiation_distribution = \
			[0.3138, 11.31, 0.3205, 108.0, 0.3307, 204.7, 0.3409, 276.0, 0.3544, 337.1, 0.3679, 393.2, 
			0.3713, 428.8, 0.3883, 469.7, 0.395, 551.1, 0.3984, 647.8, 0.4086, 790.2, 0.4086, 836.0, 
			0.4221, 871.7, 0.4323, 866.7, 0.4391, 953.3, 0.4492, 1121, 0.4526, 1187, 0.4594, 1213, 0.4695, 
			1228, 0.4797, 1264, 0.4865, 1238, 0.5, 1249, 0.5102, 1229, 0.5169, 1208, 0.5271, 1224, 0.5474, 
			1183, 0.5542, 1178, 0.5609, 1168, 0.5711, 1183, 0.5813, 1189, 0.5948, 1179, 0.6151, 1164, 0.6422, 
			1154, 0.6591, 1139, 0.6795, 1118, 0.6896, 1103, 0.6998, 1058, 0.7065, 1083, 0.7133, 1068, 0.7167, 
			915.4, 0.7180, 1027, 0.7200, 981.5, 0.7203, 747.6, 0.7235, 864.6, 0.7269, 834.2, 0.7302, 869.8, 0.7336, 
			946.1, 0.737, 976.6, 0.7506, 997.1, 0.754, 910.7, 0.7573, 854.8, 0.7607, 798.9, 0.7641, 687.1, 0.7675, 
			839.6, 0.7709, 961.7, 0.7712, 921.0, 0.781, 956.7, 0.7946, 906.0, 0.8014, 880.7, 0.8110, 870.6, 0.8115, 
			784.2, 0.8117, 845.2, 0.8135, 809.6, 0.8149, 758.8, 0.8151, 677.4, 0.8155, 713.0, 0.8160, 733.4, 0.8217, 
			713.1, 0.8318, 718.3, 0.8352, 769.2, 0.8386, 799.7, 0.8488, 825.2, 0.8691, 805.1, 0.886, 779.8, 0.8928, 
			713.8, 0.8962, 642.7, 0.8995, 581.7, 0.9063, 536.0, 0.9131, 490.3, 0.9165, 531.0, 0.9165, 592.0, 0.9233, 
			536.1, 0.9266, 495.5, 0.9266, 429.4, 0.93, 368.4, 0.9334, 276.9, 0.9334, 190.5, 0.9402, 256.7, 0.9436, 282.1, 
			0.9537, 251.7, 0.9571, 307.7, 0.9605, 378.9, 0.9639, 450.1, 0.9707, 501.0, 0.9774, 557.0, 0.9842, 618.1, 1.001, 
			592.9, 1.025, 552.4, 1.059, 496.8, 1.093, 451.4, 1.099, 466.8, 1.103, 421.0, 1.106, 375.3, 1.109, 334.7, 1.113, 
			268.6, 1.12, 172.0, 1.13, 111.1, 1.143, 146.9, 1.16, 253.8, 1.17, 320.0, 1.174, 350.6, 1.187, 360.9, 1.191, 391.4, 
			1.218, 361.2, 1.235, 366.4, 1.262, 310.8, 1.279, 321.1, 1.292, 311.1, 1.309, 275.7, 1.319, 224.9, 1.33, 169.1, 1.35, 
			113.4, 1.367, 57.61, 1.38, 6.898, 1.428, 32.8, 1.455, 68.67, 1.468, 63.73, 1.489, 109.7, 1.516, 171.0, 1.54, 227.2,
			1.567, 217.3, 1.587, 212.4, 1.607, 202.4, 1.628, 202.6, 1.651, 187.6, 1.678, 167.6, 1.702, 162.7, 1.736, 152.9, 
			1.773, 132.9, 1.797, 102.7, 1.831, 52.16, 1.858, 11.75, 1.946, 38.07, 1.99, 58.86, 2.014, 74.35, 2.051, 74.73, 
			2.088, 75.11, 2.122, 75.46, 2.149, 70.65, 2.22, 71.37, 2.295, 72.13] # Solar Irradiance [W/m ** 2/µm]
		solar_spectrum_radiation_distribution = np.reshape(solar_spectrum_radiation_distribution,[len(solar_spectrum_radiation_distribution)//2,2])    
		f = interpolate.interp1d(solar_spectrum_radiation_distribution.T[0], solar_spectrum_radiation_distribution.T[1])
		wavelength_um = np.arange(np.min(solar_spectrum_radiation_distribution.T[0]), np.max(solar_spectrum_radiation_distribution.T[0]), 0.001)
		spectrum = f(wavelength_um) # interpolation resulting spectrum with scipy.interpolate
		return spectrum, wavelength_um

	def light_heat_dissipation(self, light=None):
		return light.Voltage_V * light.Current_A - light.PeakPower_W

	def light_conversion_efficiency(self, light=None):
		return light.PeakPower_W / (light.Voltage_V * light.Current_A)

	def wallFlux(self, light=None, scene=None, dist_vec=None, lens=None, light_type='ir_pulse'):
		if light is None:
			light = self.light
		if lens is None:
			lens = self.lens
		if scene is None:
			scene = self.scene	
		if dist_vec is None:
			dist_m = scene.Distance_m
		else:
			dist_m = dist_vec

		# print(' ## wallFlux ## \n', light.PeakPower_W,'\n =====  \n' )
		# Flux on a wall (scene) during lighting time (pulse time)
		# Flux due to point source attached to the sensor
		if light_type == 'ir_pulse':
			flux = light.PeakPower_W * light.Transmission * light.Number_units / \
				(np.radians(light.Hfov_deg) * np.radians(light.Vfov_deg) * dist_m ** 2)  # [W/m^2]
		if light_type == 'solar':
			y, x = self.solarSpectrum_W_m2_um()
			### TODO: Consider replace interpolation by Numpy.Interp
			f = interpolate.interp1d(x, y)
			amb = scene.AmbientLight_lux / maximum_amb_light_lux
			if len(dist_m.shape):
				### TODO: This required due to bug in the scipy.interpolation
				solar_flux_W_m2_um = amb * f([light.WaveLength_nm / 1000])# solar flux input in [um]
				solar_flux_W_m2_nm_vec = np.array(dist_m.shape[0] * list(solar_flux_W_m2_um)) \
					* lens.BP_filter_width_nm * 0.001 # [W/m^2/um][nm][nm/um] ==> [W/m^2] 
			else:
				solar_flux_W_m2_um = amb * f(light.WaveLength_nm / 1000) # solar flux input in [um]
				solar_flux_W_m2_nm_vec = solar_flux_W_m2_um * lens.BP_filter_width_nm * 0.001 # [W/m^2/um][nm][nm/um] ==> [W/m^2] 

			flux = solar_flux_W_m2_nm_vec 
		return flux

	def siliconFlux(self, wall_flux, lens=None, scene=None):
		if lens is None:
			lens = self.lens
		if scene is None:
			scene = self.scene			
		# Flux on silicon from the scene during lighting time (pulse time)
		# Flux due to point source attached to the sensor
		# flux_active = light['peak'] * light['transmission'] / (light['hfov'] * light['vfov'] * scene['distance'] ** 2)  # [W/m^2]
		return wall_flux * scene.Reflectivity * lens.Transmission / (4 * lens.F_num ** 2)   # [W/m^2]

	def photoelectron(self, siliconFlux, sensor=None, light=None, op=None, light_type='ir_pulse'):
		if sensor is None:
			sensor = self.sensor
		if light is None:
			light = self.light
		if op is None:
			op = self.op

		if light_type == 'ir_pulse':		
			integration_time_sec = op.InBurstDutyCycle * op.BurstTime_s 
		if light_type == 'solar':
			### TODO: need to formulate integration time for solar
			integration_time_sec = op.InBurstDutyCycle * op.BurstTime_s  

		energy_to_pe = hc / (light.WaveLength_nm * nm) # Conversion from energy [J] to number of photons
		# PE on silicon from the scene during lighting time (pulse time)
		pe_per_sec = siliconFlux * (sensor.PixelSize_m ** 2) * sensor.QE * sensor.FF / energy_to_pe  # [photoelectrons / sec] 
		return pe_per_sec * integration_time_sec # [photoelectrons within integration time]

	def siliconFlux2(self, light=None, scene=None, lens=None, dist_vec=None):
		return self.siliconFlux(wall_flux=self.wallFlux(dist_vec=dist_vec))

	def photoelectron2(self, light=None, scene=None, lens=None, sensor=None, op=None, dist_vec=None):
		return self.photoelectron(siliconFlux=self.siliconFlux(wall_flux=self.wallFlux(dist_vec=dist_vec)))

	def signal_to_noise_ratio(self, sensor=None, op=None, dist_vec=None):
		T_kelvin = 300
		if sensor is None:
			sensor = self.sensor
		if op is None:
			op = self.op

		### TODO: Find appropriate integration time foe dark signal
		integration_time_sec = op.InBurstDutyCycle * op.BurstTime_s

		signal = dict(
			ir_pulse=self.photoelectron2(self, dist_vec=dist_vec),
			solar=self.photoelectron(
				siliconFlux=self.siliconFlux(
				wall_flux=self.wallFlux(dist_vec=dist_vec,light_type='solar'))))
		# print(signal)
		noise = dict(
			noise_photon_shot = (signal['ir_pulse'] + signal['solar']) ** 0.5,
			dark_noise = (self.sensor.DarkSignal_e_s * integration_time_sec)**0.5,
			readout_noise = self.sensor.ReadoutNoise_e,
			kTC_noise = 1/e_minus * np.sqrt(k_b * T_kelvin * self.sensor.Cfd_farad) if not self.sensor.CDS else 0.0, 
						# 1/[coul]*sqrt([V][A][Sec][A][Sec]/[V]) == [1] 
			quantization_noise = 0.0) ### TODO: quantizatio noise definition
		SNR = signal['ir_pulse'] / (np.sqrt(
			noise['noise_photon_shot']**2 + 
			noise['dark_noise']**2 + 
			noise['readout_noise']**2 + 
			noise['kTC_noise']**2 + 
			noise['quantization_noise']**2))
		return signal, noise, SNR

	def generate_pulse(self, delay=None, rise=None, fall=None, width=None, time_interval=None, smooth=False, mode='charge_discharge'):
		'''
		Create a light pulse (similar to a capacitor charge/discharge)
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
		return pixel_fill_factor * silicon_responsivity_a_w * hc / (wavelength_m * e_minus) # [ratio] 

	def qe_by_absorption(self, silicon_absorption_cm, pixel_fill_factor, epi_thickness_m):
		''' 
		Silicon absorption [1/CM] (changes with wavelength 2000@940nm, 500@850nm)
		fill factor [ratio]
		Epitaxial layer thickness [m]
		[1/cm]*[cm/m]*[m] ==> [1]
		'''
		cm_m = 100 # [m/cm]
		return pixel_fill_factor * (1.0 - np.exp(-silicon_absorption_cm * cm_m * epi_thickness_m)) # [ratio] 

if __name__ == '__main__':
	import os
	CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

	config = pd.read_excel(CURRENT_DIR+'/../data/photonic_simul_data.xlsx',sheet_name='Config',header=1,index_col='Name').loc['Cfg1']
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

	alpha = 500 # [1/cm] absorption coef @ 850nm
	silicon_epi = 10e-6 # [m] 
	tmp = photonic.qe_by_absorption(pixel_fill_factor=1, silicon_absorption_cm=alpha, epi_thickness_m=silicon_epi)
	print(f'QE of {1e6*silicon_epi}um EPI thickness @850 nm is {100*tmp:2.0f}%')
	alpha = 200 # [1/cm] absorption coef @ 940nm
	tmp = photonic.qe_by_absorption(pixel_fill_factor=1, silicon_absorption_cm=alpha, epi_thickness_m=silicon_epi)
	print(f'QE of {1e6*silicon_epi}um EPI thickness @940 nm is {100*tmp:2.0f}%')	
	alpha = 10000 # [1/cm] absorption coef Ge @ 1350nm
	silicon_epi = 3e-6 # [m] 
	tmp = photonic.qe_by_absorption(pixel_fill_factor=1, silicon_absorption_cm=alpha, epi_thickness_m=silicon_epi)
	print(f'QE of {1e6*silicon_epi}um EPI thickness @1350 nm is {100*tmp:2.0f}%')

	print('=====\nWall flux = {:2.3} W/m**2\nSilicon flux = {:2.3}  W/m**2\nPhotoelectron = {:5.5} photonelectron/burst\n======'.format(
		photonic.wallFlux(light_type='solar'), 
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
	
	photonic = Photonic(config='fake_tof_day_1375')
	print(f'Solar= {photonic.wallFlux(light_type="solar", dist_vec=np.array([1,2]))}')

	signal, noise, SNR = photonic.signal_to_noise_ratio(dist_vec=np.array([1,2]))
	print('====signal_to_noise_ratio ====\nSignal=', signal)
	print('Noise=', noise)
	print('SNR=', SNR)

	
	import matplotlib.pyplot as plt
	plt.plot(t1,y1)
	plt.plot(t2,y2)
	plt.plot(t3, y3)
	plt.show()

	
	

