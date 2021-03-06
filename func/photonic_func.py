import numpy as np
import pandas as pd
# import pathlib
import sys, os
# os.environ['PATH'] = "path-to-openslide-bin" + ";" + os.environ['PATH']
# from scipy.interpolate import interp1d
# from scipy import interpolate
# import scipy.signal as sp_signal

c = 2.99792458e8  # [m/s] Speed of light
hc = 1.987820871E-025  # [J * m / photon] Energy of photon with wavelength m
efficacy = 683 # [lumen/watt @ 550nm]
k_b = 1.3806488e-23 # [J/K] boltzman constant
h   = 6.62606957e-34 # [J*sec] Plank constant
e_minus =1.60217662e-19  # [Coulomb]
T_kelvin = 300
maximum_amb_light_lux = 120000 # [lux]
nm = 1e-9 # [m]
cm = 1e-2 # [m]
um = 1e-6 # [m]
nsec = 1e-9 # [sec]
fF = 1e-15 # [F]

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
	def __init__(self, config=None, input_file=None):
		self.config = config
		self.input_file = input_file
		#! Unclear why the following condition is required and not in function init
		if config is None:  
			self.config = 'Cfg1'
		
		# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) #! Mac/Linux version
		CURRENT_DIR = os.path.abspath(os.getcwd())  #! Windows version
		FUNC_DIR = os.path.split(__file__)[0]
		# _data_file = CURRENT_DIR+'/../data/SolarRadiationSpectrum/solar_spectrum_radiation_distribution.csv'
		_data_file = os.path.join(FUNC_DIR,'..','data','SolarRadiationSpectrum','solar_spectrum_radiation_distribution.csv')  
		# print(CURRENT_DIR, '\n', _data_file, '\n', os.path.split(__file__)[0])
		tmp = np.genfromtxt(_data_file, delimiter=',')
		self.solar_spectrum_radiation_distribution = np.vstack(tmp[1:])

		self.Absorption_Coefficient = dict()
		for sc in ['Si', 'Ge']:
			# _data_file = CURRENT_DIR+'/../data/AbsorptionData/'+sc+'_Absorption_Coefficient_cm-1.csv'
			_data_file = os.path.join(FUNC_DIR,'..','data','AbsorptionData',sc+'_Absorption_Coefficient_cm-1.csv')
			tmp = np.genfromtxt(_data_file, delimiter=',')
			self.Absorption_Coefficient[sc] = np.vstack(tmp[1:])

		# Read excel table having the following sheets: 'Light', 'Sensor', 'Scene', 'Lens', 'Op', 'Config'
		# _data_file = CURRENT_DIR+'/../data/photonic_simul_data.xlsx'
		if input_file is None:
			_data_file = os.path.join(FUNC_DIR,'..','data','photonic_simul_data.xlsx')
		else:
			_data_file = self.input_file
		
		self.light_ = pd.read_excel(_data_file,sheet_name='Light',header=1,index_col='Name')
		self.sensor_ = pd.read_excel(_data_file,sheet_name='Sensor',header=1,index_col='Name')
		self.scene_ = pd.read_excel(_data_file,sheet_name='Scene',header=1,index_col='Name')
		self.lens_ = pd.read_excel(_data_file,sheet_name='Lens',header=1,index_col='Name')
		self.op_ = pd.read_excel(_data_file,sheet_name='Op',header=1,index_col='Name')
		self.config_ = pd.read_excel(_data_file,sheet_name='Config',header=1,index_col='Name')

		self.update_photonic()

		self.wall_flux = self.wallFlux()
		self.silicon_flux = self.siliconFlux(self.wall_flux)

	def update_photonic(self):	
		try:			
			# Reduce parameters level to those appear in the config, only.
			# print(' ## Photonic ## \n', config_.loc[self.cfg],'\n =====  \n' )
			self.light = self.light_.loc[self.config_.loc[self.config , 'Light']]
			self.scene = self.scene_.loc[self.config_.loc[self.config , 'Scene']]
			self.lens = self.lens_.loc[self.config_.loc[self.config , 'Lens']]
			self.sensor = self.sensor_.loc[self.config_.loc[self.config , 'Sensor']]
			self.op = self.op_.loc[self.config_.loc[self.config , 'Op']]
		except KeyError as err: # check Excel data validity
			print('\033[91m'+'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
			print('Photonic::KeyError::Configutation \033[106m{}\033[0m\033[91m is mismatch key:\033[106m{}'.format(self.config, err)+ '\033[0m')
			sys.exit(1)

	def qe_by_responsivity(self, silicon_responsivity_a_w, pixel_fill_factor, wavelength_m):
		''' 
		Silicon responsivity [A/W]
		fill factor [ratio]
		wavelength [m]
		[A]/[W]*[J * m / photon]/[m]/[C] ==> [1]
		'''
		return pixel_fill_factor * silicon_responsivity_a_w * hc / (wavelength_m * e_minus) # [ratio] 

	def qe_by_absorption(self, silicon_absorption_cm, epi_thickness_um):
		''' 
		Semiconductor absorption [1/CM] 
			> Changes with wavelength Si 2000@940nm, 500@850nm, Ge: 6770@1375nm, 2442@1550nm
		fill factor [ratio]
		Epitaxial layer thickness [m]
		[1/cm]*[cm/m]*[m] ==> [1]
		'''
		_epi_thickness_m = epi_thickness_um * um
		_cm_m = 100 # [m/cm]
		return (1.0 - np.exp(-silicon_absorption_cm * _cm_m * _epi_thickness_m)) # [ratio] 

	def quantum_efficiency(self, semiconductor='Si', wavelength_nm=850, epi_thickness_um=10):
		# f = interp1d(self.Absorption_Coefficient[semiconductor].T[0], 
		#                          self.Absorption_Coefficient[semiconductor].T[1])
		# _silicon_absorption_cm = f(wavelength_nm) # interpolation resulting spectrum with scipy.interpolate
		_silicon_absorption_cm = np.interp(wavelength_nm, 	self.Absorption_Coefficient[semiconductor].T[0], 
		                         							self.Absorption_Coefficient[semiconductor].T[1])
		QE = self.qe_by_absorption(_silicon_absorption_cm, epi_thickness_um)
		return QE

	def solarSpectrum_W_m2_um(self):
		# NASA technical Memorandum1980
		# Solar Irradiance [W/m^2/µm]
		# https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19810016493.pdf
		# f = interp1d(self.solar_spectrum_radiation_distribution.T[0], 
		#                          self.solar_spectrum_radiation_distribution.T[1])
		wavelength_um = np.arange(np.min(self.solar_spectrum_radiation_distribution.T[0]), 
		                          np.max(self.solar_spectrum_radiation_distribution.T[0]), 0.001)
		# spectrum = f(wavelength_um) # interpolation resulting spectrum with scipy.interpolate
		spectrum = np.interp(wavelength_um, self.solar_spectrum_radiation_distribution.T[0], 
		                         			self.solar_spectrum_radiation_distribution.T[1])
		return spectrum, wavelength_um

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
			# f = interp1d(x, y)
			amb = scene.AmbientLight_lux / maximum_amb_light_lux
			if len(dist_m.shape):
				### TODO: This required due to bug in the scipy.interpolation
				# solar_flux_W_m2_um = amb * f([light.WaveLength_nm / 1000])# solar flux input in [um]
				solar_flux_W_m2_um = amb * np.interp([light.WaveLength_nm / 1000], x, y) # solar flux input in [um]
				solar_flux_W_m2_nm_vec = np.array(dist_m.shape[0] * list(solar_flux_W_m2_um)) \
					* lens.BP_filter_width_nm * 0.001 # [W/m^2/um][nm][nm/um] ==> [W/m^2] 
			else:
				# solar_flux_W_m2_um = amb * f(light.WaveLength_nm / 1000) # solar flux input in [um]
				solar_flux_W_m2_um = amb * np.interp(light.WaveLength_nm / 1000, x, y) # solar flux input in [um]
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

		# if light_type == 'ir_pulse':		
		# 	_integration_time_sec = op.InBurstDutyCycle * op.BurstTime_s 
		# if light_type == 'solar':
		# 	### TODO: need to formulate integration time for solar
		# 	_integration_time_sec = op.InBurstDutyCycle * op.BurstTime_s 
		
		_QE = self.quantum_efficiency(semiconductor=sensor.Semiconductor,
									 wavelength_nm=sensor.Wavelength,
									 epi_thickness_um=sensor.epi_thick_um)
		# print('>>>>>')
		# print(sensor, '\nQE->', _QE)
		# print('<<<<<')
		_energy_to_pe = hc / (light.WaveLength_nm * nm) # Conversion from energy [J] to number of photons
		# PE on silicon from the scene during lighting time (pulse time)
		pe_per_sec = siliconFlux * ((sensor.PixSz_um * um) ** 2) * _QE * sensor.FF / _energy_to_pe  # [photoelectrons / sec] 
		return pe_per_sec * op.integTime_s # [photoelectrons within integration time]

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

		# ### TODO: Find appropriate integration time for dark signal
		# _integration_time_sec = op.InBurstDutyCycle * op.BurstTime_s

		signal = dict(
			ir_pulse=self.photoelectron2(self, dist_vec=dist_vec),
			solar=self.photoelectron(
				siliconFlux=self.siliconFlux(
				wall_flux=self.wallFlux(dist_vec=dist_vec,light_type='solar'))))
		# print(signal)
		noise = dict(
			noise_photon_shot = (signal['ir_pulse'] + signal['solar']) ** 0.5,
			dark_noise = (sensor.DkSig_e_s * op.integTime_s) ** 0.5,
			readout_noise = sensor.RdNoise_e,
			kTC_noise = 1/e_minus * np.sqrt(k_b * T_kelvin * sensor.Cfd_fF * fF) if not self.sensor.CDS else 0.0, 
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
			x_fall = np.arange(0, fall * 5, self.time_interval) # Discharge takes longer to become zero
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
			f = np.hamming(15) 
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
	
	def generate_tof_shutters(self, op, freq, shutter_per_cycle, total_shutters):
		sh_vector_length = 100
		cycle_time = 1 / freq

		# Get light + shutter pulse data
		light_rise = op.light_rise_nsec * nsec
		light_fall = op.light_fall_nsec * nsec
		light_width = op.light_width_nsec * nsec
		shutter_rise = op.shutter_rise_nsec * nsec
		shutter_fall = op.shutter_fall_nsec * nsec
		shutter_width = op.shutter_width_nsec * nsec
		shutter_delay = op.shutter_delay_nsec * nsec

		# Generate light pulse, shutter pulse, and a convolution
		y1, t1 = self.generate_pulse(rise=light_rise, fall=light_fall, width=light_width, smooth=True)
		y2, t2 = self.generate_pulse(delay=shutter_delay, rise=shutter_rise, fall=shutter_fall, width=shutter_width, smooth=True)
		y3, t3 = self.conv_light_shutter(t_light=t1, y_light=y1, t_shutter=t2, y_shutter=y2)

		# Generates the raw shutters by applying the sh_delay (the will not overlap in between)
		shutters = dict()
		for i in range(total_shutters):
			sh_delay = i * cycle_time / shutter_per_cycle
			shutters['sh_'+str(i)] = np.stack((t3 + sh_delay, y3)).T

		# New time shift axis
		shutters['dt'] = np.linspace(shutters['sh_0'][:,0].min(), shutters['sh_'+str(total_shutters - 1)][:,0].max(), sh_vector_length)

		# Interpolate the raw shutters into the new time shift axis
		for i in range(total_shutters):
			shutters['sh'+str(i)]= np.interp(shutters['dt'], shutters['sh_'+str(i)][:,0], shutters['sh_'+str(i)][:,1])
			del shutters['sh_'+str(i)] # Delete the raw shutters
		
		sh_assign = list(np.array(list(range(total_shutters))) % shutter_per_cycle)

		for i  in range(shutter_per_cycle):
			shutters['sh_comb'+str(i)] = shutters['sh'+str(i)]
			if total_shutters > shutter_per_cycle:
				for j, k in enumerate(sh_assign):
					if j >= shutter_per_cycle and k == i:
						shutters['sh_comb'+str(i)] = shutters['sh_comb'+str(i)] + shutters['sh'+str(j)]
		return shutters

	def get_modulation_phase(self, shutters, shutter_per_cycle, initial_phase = -3.0, key='sh_comb'):
		# Initialize phase components
		XX = np.zeros(shutters['sh0'].shape)
		YY = np.zeros(shutters['sh0'].shape)
		
		# Extract phase components from the phasors
		for i in range(shutter_per_cycle):
			XX = shutters[key+str(i)] * np.cos(2*np.pi*i/shutter_per_cycle + initial_phase) + XX
			YY = shutters[key+str(i)] * np.sin(2*np.pi*i/shutter_per_cycle + initial_phase) + YY

			# Unwrap the phase
		phase = np.arctan2(XX, YY)
		return phase

	def power_consumption(self, light=None, op=None, sensor=None):
		if light is None:
			light = self.light
		if op is None:
			op = self.op
		if sensor is None:
			sensor = self.sensor
		
		_light_duty_cycle = op.integTime_s / (1 / op.frame_rate)
		power = dict()
		power['optical'] = light.PeakPower_W * _light_duty_cycle * light.Number_units
		power['light'] = light.Voltage_V * light.Current_A * _light_duty_cycle * light.Number_units
		power['sensor'] = sensor.Power_nJ_pix * 1e-9 * sensor.Res_H * sensor.Res_H * op.frame_rate * op.N_subframes 
		power['heat_light'] = power['light'] * (light.Voltage_V * light.Current_A - light.PeakPower_W) / light.PeakPower_W
		return power



if __name__ == '__main__':
	# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
	CURRENT_DIR = os.path.abspath(os.getcwd())  #! Windows version
	FUNC_DIR = os.path.split(__file__)[0]
	# config = pd.read_excel(CURRENT_DIR+'/../data/photonic_simul_data.xlsx',sheet_name='Config',header=1,index_col='Name').loc['Cfg1']
	
	_data_file = os.path.join(FUNC_DIR,'..','data','photonic_simul_data.xlsx')
	config = pd.read_excel(_data_file,sheet_name='Config',header=1,index_col='Name').loc['Cfg1']
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
	silicon_epi = 1e-6 # [m] 
	tmp = photonic.qe_by_absorption(silicon_absorption_cm=alpha, epi_thickness_um=silicon_epi)
	print(f'QE of {silicon_epi}um EPI thickness @850 nm is {100*tmp:2.0f}%')
	alpha = 200 # [1/cm] absorption coef @ 940nm
	tmp = photonic.qe_by_absorption(silicon_absorption_cm=alpha, epi_thickness_um=silicon_epi)
	print(f'QE of {silicon_epi}um EPI thickness @940 nm is {100*tmp:2.0f}%')	
	alpha = 10000 # [1/cm] absorption coef Ge @ 1350nm
	silicon_epi = 3e-6 # [m] 
	tmp = photonic.qe_by_absorption(silicon_absorption_cm=alpha, epi_thickness_um=silicon_epi)
	print(f'QE of {1e6*silicon_epi}um EPI thickness @1350 nm is {100*tmp:2.0f}%')

	print('=====\nWall flux = {:2.3} W/m**2\nSilicon flux = {:2.3}  W/m**2\nPhotoelectron = {:5.5} photonelectron/burst\n======'.format(
		photonic.wallFlux(light_type='solar'), 
	    photonic.siliconFlux(wall_flux=photonic.wallFlux()),
	    photonic.photoelectron(siliconFlux=photonic.siliconFlux(wall_flux=photonic.wallFlux())))) #,'\n',photonic.photoelectron())	

	
	photonic = Photonic(config='fake_tof_day_1375')
	print(f'Solar= {photonic.wallFlux(light_type="solar", dist_vec=np.array([1,2]))}')

	signal, noise, SNR = photonic.signal_to_noise_ratio(dist_vec=np.array([1,2]))
	print('====signal_to_noise_ratio ====\nSignal=', signal)
	print('Noise=', noise)
	print('SNR=', SNR)

	
	# photonic = Photonic(config='Cfg3')
	# rise = photonic.op_.loc[photonic.config_.loc[photonic.config,'Op'],'light_rise_nsec']
	# fall = photonic.op_.loc[photonic.config_.loc[photonic.config,'Op'],'light_fall_nsec']
	# width = photonic.op_.loc[photonic.config_.loc[photonic.config,'Op'],'light_width_nsec']
	# y1, t1 = photonic.generate_pulse(rise=rise, fall=fall, width=width, smooth=False)


	# rise = photonic.op_.loc[photonic.config_.loc[photonic.config,'Op'],'shutter_rise_nsec']
	# fall = photonic.op_.loc[photonic.config_.loc[photonic.config,'Op'],'shutter_fall_nsec']
	# width = photonic.op_.loc[photonic.config_.loc[photonic.config,'Op'],'shutter_width_nsec']
	# delay = photonic.op_.loc[photonic.config_.loc[photonic.config,'Op'],'shutter_delay_nsec']
	# delay = 14e-9
	# width=5e-9
	# y2, t2 = photonic.generate_pulse(delay=delay, rise=rise, fall=fall, width=width, smooth=False)
	# y3, t3 = photonic.conv_light_shutter(t_light=t1, y_light=y1, t_shutter=t2, y_shutter=y2)

	# import matplotlib.pyplot as plt
	# fig, ax = plt.subplots(3,1, sharex=True)
	# ax[0].plot(t1,y1, label='Light')
	# ax[0].set_ylabel('Light')
	# ax[0].grid()
	# ax[1].plot(t2,y2, label='Shutter')
	# ax[1].set_ylabel('Shutter')
	# ax[1].grid()
	# ax[2].plot(t3, y3, label='Convolution')
	# ax[2].set_ylabel('Convolution')
	# ax[2].grid()
	# ax[2].set_ylim(0,1)
	# ax[2].text(0,0.1,'Convolution units are fraction of the integrated light')
	# plt.show()

	photonic = Photonic(config='Cfg3')
	freq = 60e6
	shutter_per_cycle = 3
	total_shutters = shutter_per_cycle + 2

	shutters = photonic.generate_tof_shutters(op=photonic.op, freq=freq, 
									shutter_per_cycle=shutter_per_cycle, 
									total_shutters=total_shutters)                                                                           
	print('shutters: ',shutters['dt'].shape, shutters['sh0'].shape)

	power = photonic.power_consumption()
	P_light = power['light'] * 1000 # [mW]
	P_sensor = power['sensor'] * 1000 # [mW]
	P_heat = power['heat_light'] * 1000 + P_sensor
	P_total = P_light + P_sensor  
	print(f'Light Power Consumption is {P_light:2.1f} mW\nSensor Power Consumption is {P_sensor:2.1f} mW\nTotal Power Consumption is {P_total:2.1f} mW')
	print(f'Total Heat Dissipation is {P_heat:2.1f} mW')

	filename = '../sys_arch/data/photonic_simul/photonic_simul_data.xlsx'
	# filename = r'C:\Users\david.cohen\repo\imaging\data\photonic_simul_data.xlsx'
	photonic = Photonic(input_file=filename)

