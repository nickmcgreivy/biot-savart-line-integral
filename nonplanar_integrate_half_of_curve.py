import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from numpy.random import rand

def biot_savart_hanson_hirshman(r_eval, l, I = 1.0, wrap=True):
	"""
	Piecewise linear approach
	l are points along measurement

	If "wrap" is true, it means l is a length NSx3 vector, with NS distinct points,
	and the 0th point is connected to the last point. It is a closed loop.

	If "wrap" is false, we treat the 0th point as being not connected to the last point.
	This happens, for example, in computing the field from the curved section of
	the D-shaped coil. 
	"""
	if wrap:
		l = np.concatenate((l, l[0:1]))
	R_i_vec = r_eval[None, :] - l[:-1, :]
	R_f_vec = r_eval[None, :] - l[1:,  :]
	dl = l[1:, :] - l[:-1, :]
	L = np.linalg.norm(dl, axis=-1)
	e_hat = dl / L[:, None]
	R_i = np.linalg.norm(R_i_vec, axis=-1)
	R_f = np.linalg.norm(R_f_vec, axis=-1)
	coeff = I * 2 * L * (R_i + R_f) / (R_i * R_f) / ((R_i + R_f) ** 2 - L ** 2)
	vec = np.cross(e_hat, R_i_vec, axis=-1)
	return np.sum(vec * coeff[:, None], axis=0)
def dot(A, B):
	"""
	A helper function, dot product along last dimension of two Nx3 vectors
	"""
	return A[:, 0] * B[:, 0] +  A[:, 1] * B[:, 1] + A[:, 2] * B[:, 2]

def get_normal(p, NS):
	"""
	Returns normal vector to curve described by Fourier series.
	"""
	r2 = d2l_dt2(p, NS)
	r1 = dl_dt(p, NS)
	T = tangent(p, NS)
	r1_norm_squared = np.linalg.norm(r1, axis=-1) ** 2
	dT_ds = (r2 - T * dot(T, r2)[:, np.newaxis]) / r1_norm_squared[:, np.newaxis]
	return dT_ds / np.linalg.norm(dT_ds, axis=-1)[:, np.newaxis]


def tangent(p, NS):
	"""
	Computes the tangent vector for a curve parametrized by a Fourier series.
	"""
	deriv = dl_dt(p, NS)
	return deriv / np.linalg.norm(deriv, axis=-1)[:, np.newaxis]
	
def get_curvature(p, NS):
	"""
	Get curvature of curve described by Fourier series.
	"""
	r2 = d2l_dt2(p, NS)
	r1 = dl_dt(p, NS)
	top = np.linalg.norm(np.cross(r2, r1), axis=-1)
	bottom = np.linalg.norm(r1, axis=-1)**3
	return top / bottom 


def error_in_B(B_1, B_2): 
	""" 
	L2 norm between B1 and B2 where B1 is the magnetic field as the number of points goes to infinity.
	"""
	return np.linalg.norm((B_1 - B_2), ord=2) / np.linalg.norm(B_1, ord = 2)

def dl_dt(p, NS):
	"""
	Computes dl/dt where l is parametrized by a fourier series in t.
	"""
	#assert NS % 2 == 0
	theta = np.linspace(0.0 + 1. / (2 * NS), 1.0 - 1. / (2 * NS), NS)
	theta = theta * np.pi
	r1 = np.zeros((3, NS))
	for m in range(p.shape[1]):
		r1 += - p[:3, None, m] * m * np.sin(m * theta)[None, :] + p[3:, None, m] * m * np.cos(m * theta)[None, :]
	return np.transpose(r1, (1, 0))

def d2l_dt2(p, NS):
	"""
	Computes d^2 l/dt^2 where l is parametrized by a fourier series in t.
	"""
	#assert NS % 2 == 0
	theta = np.linspace(0.0 + 1. / (2 * NS), 1.0 - 1. / (2 * NS), NS)
	theta = theta * np.pi
	r2 = np.zeros((3, NS))
	for m in range(p.shape[1]):
		r2 += - p[:3, None, m] * m**2 * np.cos(m * theta)[None, :] - p[3:, None, m] * m**2 * np.sin(m * theta)[None, :]
	return np.transpose(r2, (1, 0))

def fourier_to_half_real_space(p, NS):
	theta = np.linspace(0.0 + 1. / (2 * NS), 1.0 - 1. / (2 * NS), NS)
	theta = np.concatenate((np.asarray([0.0]), theta, np.asarray([1.0])), axis=0)
	theta = theta * np.pi
	r = np.zeros((3, NS+2))
	for m in range(p.shape[1]):
		r += p[:3, None, m] * np.cos(m * theta)[None, :] + p[3:, None, m] * np.sin(m * theta)[None, :]
	r = np.transpose(r, (1, 0))
	return r

def fourier_theta(p, theta):
	r = np.zeros(3)
	for m in range(p.shape[1]):
		r += p[:3, m] * np.cos(m * theta) + p[3:, m] * np.sin(m * theta)
	return r

def l_shift_half_curvature(p, NS):
	"""
	Computes the shifted points starting with the Fourier series.
	"""
	l_original = fourier_to_half_real_space(p, NS)
	l_start = l_original[0]
	l_end = l_original[-1]
	l = l_original[1:-1]
	normal = get_normal(p, NS)
	curvature = get_curvature(p, NS)
	dt = np.pi / NS
	L = np.linalg.norm(dl_dt(p, NS) * dt, axis=-1)

	l_final = l - (curvature * L**2)[:, np.newaxis] * normal / 12
	return np.concatenate([l_start[None, :], l_final, l_end[None, :]], axis=0)


########
# APPENDIX D FUNCTIONS
########

def get_normal_D(p, NS):
	"""
	Returns normal vector to curve described by Fourier series.
	"""
	r2 = d2l_dt2_D(p, NS)
	r1 = dl_dt_D(p, NS)
	T = tangent_D(p, NS)
	r1_norm_squared = np.linalg.norm(r1, axis=-1) ** 2
	dT_ds = (r2 - T * dot(T, r2)[:, np.newaxis]) / r1_norm_squared[:, np.newaxis]
	return dT_ds / np.linalg.norm(dT_ds, axis=-1)[:, np.newaxis]


def tangent_D(p, NS):
	"""
	Computes the tangent vector for a curve parametrized by a Fourier series.
	"""
	deriv = dl_dt_D(p, NS)
	return deriv / np.linalg.norm(deriv, axis=-1)[:, np.newaxis]
	
def get_curvature_D(p, NS):
	"""
	Get curvature of curve described by Fourier series.
	"""
	r2 = d2l_dt2_D(p, NS)
	r1 = dl_dt_D(p, NS)
	top = np.linalg.norm(np.cross(r2, r1), axis=-1)
	bottom = np.linalg.norm(r1, axis=-1)**3
	return top / bottom 

def dl_dt_D(p, NS):
	"""
	Computes dl/dt where l is parametrized by a fourier series in t.
	"""
	ds = 1/(NS - 2 + np.sqrt(2))
	ds_prime = ds/np.sqrt(2)
	theta = np.linspace(ds_prime, 1.0 - ds_prime, NS-1)
	theta = theta * np.pi
	r1 = np.zeros((3, NS-1))
	for m in range(p.shape[1]):
		r1 += - p[:3, None, m] * m * np.sin(m * theta)[None, :] + p[3:, None, m] * m * np.cos(m * theta)[None, :]
	return np.transpose(r1, (1, 0))

def d2l_dt2_D(p, NS):
	"""
	Computes d^2 l/dt^2 where l is parametrized by a fourier series in t.
	"""
	ds = 1/(NS - 2 + np.sqrt(2))
	ds_prime = ds/np.sqrt(2)
	theta = np.linspace(ds_prime, 1.0 - ds_prime, NS-1)
	theta = theta * np.pi
	r2 = np.zeros((3, NS-1))
	for m in range(p.shape[1]):
		r2 += - p[:3, None, m] * m**2 * np.cos(m * theta)[None, :] - p[3:, None, m] * m**2 * np.sin(m * theta)[None, :]
	return np.transpose(r2, (1, 0))


def fourier_to_half_real_space_D(p, NS):
	ds = 1/(NS - 2 + np.sqrt(2))
	ds_prime = ds/np.sqrt(2)
	theta = np.linspace(ds_prime, 1.0 - ds_prime, NS-1)
	theta = np.concatenate((np.asarray([0.0]), theta, np.asarray([1.0])), axis=0)
	theta = theta * np.pi
	r = np.zeros((3, NS+1))
	for m in range(p.shape[1]):
		r += p[:3, None, m] * np.cos(m * theta)[None, :] + p[3:, None, m] * np.sin(m * theta)[None, :]
	r = np.transpose(r, (1, 0))
	return r

def l_shift_half_curvature_D(p, NS):
	"""
	Computes the shifted points starting with the Fourier series.
	"""
	l_original = fourier_to_half_real_space_D(p, NS)
	l_start = l_original[0]
	l_end = l_original[-1]
	l = l_original[1:-1]
	normal = get_normal_D(p, NS)
	curvature = get_curvature_D(p, NS)
	ds = 1/(NS - 2 + np.sqrt(2))
	dt = np.pi * ds
	L = np.linalg.norm(dl_dt_D(p, NS) * dt, axis=-1)
	l_final = l - (curvature * L**2)[:, np.newaxis] * normal / 12
	return np.concatenate([l_start[None, :], l_final, l_end[None, :]], axis=0)

p = np.load("w7x_fc.npy")[:, 2, :] # 2nd coil
# 10 random measurement points
R = 3

NSs = np.asarray([10, 12, 14, 16, 20, 24, 30, 40, 50, 60, 100, 200, 400, 800, 1600, 3200, 6400, 10000])

# Where I will store the errors
hanson_hirshman = np.zeros(NSs.shape)
hanson_hirshman_shift = np.zeros(NSs.shape)
hanson_hirshman_shift_D = np.zeros(NSs.shape)

N_plot = 10
l_plot = fourier_to_half_real_space(p, N_plot)
l_plot_shift = l_shift_half_curvature(p, N_plot)
l_plot_shift_D = l_shift_half_curvature_D(p, N_plot)
plt.scatter(l_plot[:,1], l_plot[:,2], color='blue')
plt.scatter(l_plot_shift[:,1], l_plot_shift[:,2], color='red')
plt.scatter(l_plot_shift_D[:,1], l_plot_shift_D[:,2], color='green')
plt.show()

for r in range(R):
	rand_vec = (rand((3)) - 0.5) * 0.4

	##################################################################
	# W7-X Coil
	##################################################################


	max_NS = int(1e6)

	# Measurement point slightly off-axis of coil
	r_eval = np.asarray([5.4, 1.9, -0.3]) + rand_vec
	l = fourier_to_half_real_space(p, max_NS)
	B_exact = biot_savart_hanson_hirshman(r_eval, l, wrap=False)

	error_func = partial(error_in_B, B_exact)


	for i in range(NSs.shape[0]):
		NS = int(NSs[i])
		print(NS)

		l = fourier_to_half_real_space(p, NS)

		# hanson and hirshman
		hanson_hirshman[i] += (error_func(biot_savart_hanson_hirshman(r_eval, l, wrap=False)))

		# shift
		l_shift = l_shift_half_curvature(p, NS)
		hanson_hirshman_shift[i] += (error_func(biot_savart_hanson_hirshman(r_eval, l_shift, wrap=False)))

		# shift, appendix D
		l_shift_D = l_shift_half_curvature_D(p, NS)
		hanson_hirshman_shift_D[i] += (error_func(biot_savart_hanson_hirshman(r_eval, l_shift_D, wrap=False)))


font = {'size'   : 12}

plt.rc('font', **font)
plt.loglog(NSs, hanson_hirshman / R, label = "Standard piecewise linear")
plt.scatter(NSs, hanson_hirshman / R)
plt.loglog(NSs, hanson_hirshman_shift / R, label = "Shifted piecewise linear")
plt.scatter(NSs, hanson_hirshman_shift / R)
plt.loglog(NSs, hanson_hirshman_shift_D / R, label = "Shifted piecewise linear, Appendix D")
plt.scatter(NSs, hanson_hirshman_shift_D / R)
plt.title("Field Error, Non-Planar Coil")
plt.xlabel("Number of discretization points")
plt.ylabel("Normalized error in magnetic field")
plt.xlim((6.5,15000))
plt.ylim((1e-11,1e-1))
plt.legend()
plt.grid(True, which="both")
plt.show()