import numpy as np
from functools import partial
import matplotlib.pyplot as plt


def biot_savart(r_eval, l, dl, I = 1.0):
	"""
	Continuous tangent approach
	r_eval is length-3 vector, l and dl are NS x 3 vectors
	"""
	r_minus_l = r_eval[None, :] - l
	top = I * np.cross(dl, r_minus_l / np.linalg.norm(r_minus_l, axis=-1)[:, None], axis=-1)
	bottom = np.linalg.norm(r_minus_l, axis=-1) ** 2
	return np.sum(top / bottom[:, None], axis = 0)

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

def fourier_to_real_space(p, NS, n = 1):
	"""
	NOTE: NS needs to be even

	n=2 is the second-order quadrature with NS/2 nodes and NS points. 
	"""

	assert NS % 2 == 0
	theta = np.linspace(0.0 + 1. / (2 * NS), 1.0 - 1. / (2 * NS), NS)
	if n == 2:
		theta[::2] += 1 / NS * (1/2 - 1/np.sqrt(3))
		theta[1::2] -= 1 / NS * (1/2 - 1 / np.sqrt(3))
	theta = theta * 2 * np.pi
	r = np.zeros((3, NS))
	for m in range(p.shape[1]):
		r += p[:3, None, m] * np.cos(m * theta)[None, :] + p[3:, None, m] * np.sin(m * theta)[None, :]
	return np.transpose(r, (1, 0))


def dl_dt(p, NS, n = 1):
	"""
	Computes dl/dt where l is parametrized by a fourier series in t.
	"""
	assert NS % 2 == 0
	theta = np.linspace(0.0 + 1. / (2 * NS), 1.0 - 1. / (2 * NS), NS)
	if n == 2:
		theta[::2] += 1 / NS * (1/2 - 1/np.sqrt(3))
		theta[1::2] -= 1 / NS * (1/2 - 1 / np.sqrt(3))
	theta = theta * 2 * np.pi
	r1 = np.zeros((3, NS))
	for m in range(p.shape[1]):
		r1 += - p[:3, None, m] * m * np.sin(m * theta)[None, :] + p[3:, None, m] * m * np.cos(m * theta)[None, :]
	return np.transpose(r1, (1, 0))

def d2l_dt2(p, NS):
	"""
	Computes d^2 l/dt^2 where l is parametrized by a fourier series in t.
	"""
	assert NS % 2 == 0
	theta = np.linspace(0.0 + 1. / (2 * NS), 1.0 - 1. / (2 * NS), NS)
	theta = theta * 2 * np.pi
	r2 = np.zeros((3, NS))
	for m in range(p.shape[1]):
		r2 += - p[:3, None, m] * m**2 * np.cos(m * theta)[None, :] - p[3:, None, m] * m**2 * np.sin(m * theta)[None, :]
	return np.transpose(r2, (1, 0))


def tangent(p, NS):
	"""
	Computes the tangent vector for a curve parametrized by a Fourier series.
	"""
	deriv = dl_dt(p, NS)
	return deriv / np.linalg.norm(deriv, axis=-1)[:, np.newaxis]


def l_shiftcurvature(p, NS):
	"""
	Computes the shifted points starting with the Fourier series.
	"""
	l_original = fourier_to_real_space(p, NS)
	normal = get_normal(p, NS)
	curvature = get_curvature(p, NS)
	dt = np.pi * 2 / NS
	L = np.linalg.norm(dl_dt(p, NS) * dt, axis=-1)
	return l_original - (curvature * L**2)[:, np.newaxis] * normal / 12

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