import numpy as np
import pandas as pd
import sys

class Photonic:
	def __init__(self, config=None):
		self.c = 2.99792458e8  # [m/s] Speed of light
		self.hc = 1.987820871E-025  # [J * m / photon] Energy of photon with wavelength m
		self.nm = 1e-9
		self.config = config
		if config is None:
			self.config = 'Cfg1'

		# Read excel table 
		self.light_ = pd.read_excel('../data/photonic_simul_data.xlsx',sheet_name='Light',header=1,index_col='Name')
		self.sensor_ = pd.read_excel('../data/photonic_simul_data.xlsx',sheet_name='Sensor',header=1,index_col='Name')
		self.scene_ = pd.read_excel('../data/photonic_simul_data.xlsx',sheet_name='Scene',header=1,index_col='Name')
		self.lens_ = pd.read_excel('../data/photonic_simul_data.xlsx',sheet_name='Lens',header=1,index_col='Name')
		self.op_ = pd.read_excel('../data/photonic_simul_data.xlsx',sheet_name='Op',header=1,index_col='Name')
		self.config_ = pd.read_excel('../data/photonic_simul_data.xlsx',sheet_name='Config',header=1,index_col='Name')

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
			# raise
			sys.exit(1)


		self.wall_flux = self.wallFlux()
		self.silicon_flux = self.siliconFlux(self.wall_flux)

	def wallFlux(self, light=None, scene=None):
		if light is None:
			light = self.light
		if scene is None:
			scene = self.scene			
		# print(' ## wallFlux ## \n', light.PeakPower_W,'\n =====  \n' )
		# Flux on a wall (scene) during lighting time (pulse time)
		# Flux due to point source attached to the sensor
		return light.PeakPower_W * light.Transmission / (np.radians(light.Hfov_deg) * np.radians(light.Vfov_deg) * scene.Distance_m ** 2)  # [W/m^2]

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
			
		energy_to_pe = self.hc / (light.WaveLength_nm * self.nm) # Conversion from energy [J] to number of photons
		# PE on silicon from the scene during lighting time (pulse time)
		pe_per_sec = siliconFlux * (sensor.PixelSize_m ** 2) * sensor.QE * sensor.FF / energy_to_pe  # [photons / sec] 
		pe_per_burst = pe_per_sec * op.InBurstDutyCycle * op.BurstTime_s
		return pe_per_burst

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

	# photonic = Photonic(config='Cfg2')	
	# call function wallFlux
	print('=====\nWall flux = {:2.3} W/m**2\nSilicon flux = {:2.3}  W/m**2\nPhotoelectron = {:5.5} photonelectron/burst\n======'.format(
		photonic.wallFlux(), 
	    photonic.siliconFlux(wall_flux=photonic.wallFlux()),
	    photonic.photoelectron(siliconFlux=photonic.siliconFlux(wall_flux=photonic.wallFlux())))) #,'\n',photonic.photoelectron())

	photonic = Photonic(config='Cfg2')
	print('=====\nWall flux = {:2.3} W/m**2\nSilicon flux = {:2.3}  W/m**2\nPhotoelectron = {:5.5} photonelectron/burst\n======'.format(
		photonic.wallFlux(), 
	    photonic.siliconFlux(wall_flux=photonic.wallFlux()),
	    photonic.photoelectron(siliconFlux=photonic.siliconFlux(wall_flux=photonic.wallFlux())))) #,'\n',photonic.photoelectron())	
