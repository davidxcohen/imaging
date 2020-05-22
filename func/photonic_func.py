import numpy as np

c = 2.99792458e8  # [m/s] Speed of light
hc = 1.987820871E-025  # [J m / photon] Energy of photon with wavelength m


def wallFlux(light, scene):
	return light['peak'] * light['transmission'] / (light['hfov'] * light['vfov'] * scene['distance'] ** 2)  # [W/m^2]

def siliconFlux(wallFlux, lens, scene):
	# flux_active = light['peak'] * light['transmission'] / (light['hfov'] * light['vfov'] * scene['distance'] ** 2)  # [W/m^2]
	return wallFlux * scene['reflectivity'] * lens['transmission'] / (4 * lens['f_num']**2)   # [W/m^2]

def photoelectron(siliconFlux, sensor, light):
	energy_to_pe = hc / light['wavelength']  # from [J] to number of photons
	pe_per_sec = siliconFlux * sensor['pix_size_m'] ** 2 * sensor['qe'] * sensor['ff'] / energy_to_pe  # [photons / sec] - If the laser was open continuously