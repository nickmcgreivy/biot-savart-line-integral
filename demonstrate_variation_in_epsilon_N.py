import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from biotsavart import * 
from numpy.random import rand

# number of random measurement points. You can 
# decrease this to 1 or 2 if you want to rapidly
# create plots for the purpose of visualizing what
# they look like. But change it back to 10 when 
# you're ready to save the plots for the paper. 
R = 100
max_NS = int(2e6)
NS_max = int(1e6)

fig,axes = plt.subplots(1, 3, figsize=(12, 4.5), sharex=True, sharey=True)
title_size = 16
xlabel_size = 15
ylabel_size = 14

##################################################################
# Circular Coil
##################################################################


NSs = np.asarray([10, 12, 14, 16, 20, 24, 30, 40, 50, 60, 100, 200, 400, 800, 1600, 3200, 6400, 10000])


quadrature = np.zeros(NSs.shape)
quadrature_n2 = np.zeros(NSs.shape)
hanson_hirshman_circle = np.zeros(NSs.shape)
hanson_hirshman_shift_circle = np.zeros(NSs.shape)

for r in range(R):
	print(r)
	rand_vec = (rand((3)) - 0.5)

	p = np.asarray([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
	r_eval = np.asarray([0.0, 0.5, 1.0]) + rand_vec
	l = fourier_to_real_space(p, max_NS)
	dl = dl_dt(p, max_NS) * (2 * np.pi / max_NS)
	B_analytical = biot_savart(r_eval, l, dl)
	error_func = partial(error_in_B, B_analytical)
	



	hh = np.zeros(NSs.shape)
	hh_shift = np.zeros(NSs.shape)
	for i in range(NSs.shape[0]):
		NS = int(NSs[i])
		# analytic tangent, n = 1
		l = fourier_to_real_space(p, NS)
		dl = dl_dt(p, NS) * (2 * np.pi / NS)
		quadrature[i] += (error_func(biot_savart(r_eval, l, dl)))
		# analytic tangent, n = 2
		l2 = fourier_to_real_space(p, NS, n = 2)
		dl2 = dl_dt(p, NS, n = 2) * (2 * np.pi / NS)
		quadrature_n2[i] += (error_func(biot_savart(r_eval, l2, dl2)))
		# hanson and hirshman
		hh[i] = (error_func(biot_savart_hanson_hirshman(r_eval, l)))
		hanson_hirshman_circle[i] += hh[i]
		# shift
		l_shift = l_shiftcurvature(p, NS)
		hh_shift[i] = (error_func(biot_savart_hanson_hirshman(r_eval, l_shift)))
		hanson_hirshman_shift_circle[i] += hh_shift[i]

	axes[0].loglog(NSs, hh, alpha=0.25, color = 'blue', linewidth=0.5)#, label = "Shifted piecewise linear")
	axes[0].loglog(NSs, hh_shift, alpha=0.25, color = 'orange', linewidth=0.5)


##################################################################
# PLOTTING
##################################################################

#axes[0].xlim((6.5,15000))
#axes[0].ylim((1e-11,1e-1))
axes[0].loglog(NSs, hanson_hirshman_circle / R, label = "Standard piecewise linear")
axes[0].scatter(NSs, hanson_hirshman_circle / R)
axes[0].loglog(NSs, hanson_hirshman_shift_circle / R, label = "Shifted piecewise linear")
axes[0].scatter(NSs, hanson_hirshman_shift_circle / R)
axes[0].set_title("Circular Coil", fontsize=title_size)
axes[1].set_xlabel("Number of discretization points", fontsize=xlabel_size)
axes[0].set_ylabel("Normalized error in magnetic field", fontsize=ylabel_size)
axes[0].set_xlim((6.5,15000))
axes[1].set_xlim((6.5,15000))
axes[2].set_xlim((6.5,15000))
axes[0].set_ylim((1e-11,1e-1))
axes[0].grid(True, which="major")

##################################################################
# END PLOTTING
##################################################################


##################################################################
# Non-Planar Coil
##################################################################

NSs = np.asarray([10, 12, 14, 16, 20, 24, 30, 40, 50, 60, 100, 200, 400, 800, 1600, 3200, 6400, 10000])

# Where I will store the errors
quadrature = np.zeros(NSs.shape)
quadrature_n2 = np.zeros(NSs.shape)
hanson_hirshman_w7x = np.zeros(NSs.shape)
hanson_hirshman_shift_w7x = np.zeros(NSs.shape)

for r in range(R):
	print(r)
	rand_vec = (rand((3)) - 0.5) * 0.5


	p = np.load("w7x_fc.npy")[:, 2, :] # 2nd coil

	# Measurement point slightly off-axis of coil
	r_eval = np.asarray([5.4, 1.9, -0.3]) + rand_vec
	l = fourier_to_real_space(p, max_NS)
	dl = dl_dt(p, max_NS) * (2 * np.pi / max_NS)

	# Here I'm not using the analytic formula for the "exact" solution, but we could 
	B_exact = biot_savart(r_eval, l, dl)

	error_func = partial(error_in_B, B_exact)

	hh = np.zeros(NSs.shape)
	hh_shift = np.zeros(NSs.shape)
	for i in range(NSs.shape[0]):
		NS = int(NSs[i])

		# analytic tangent, n = 1
		l = fourier_to_real_space(p, NS)
		dl = dl_dt(p, NS) * (2 * np.pi / NS)
		quadrature[i] += (error_func(biot_savart(r_eval, l, dl)))

		# analytic tangent, n = 2
		l2 = fourier_to_real_space(p, NS, n = 2)
		dl2 = dl_dt(p, NS, n = 2) * (2 * np.pi / NS)
		quadrature_n2[i] += (error_func(biot_savart(r_eval, l2, dl2)))

		# hanson and hirshman
		hh[i] = (error_func(biot_savart_hanson_hirshman(r_eval, l)))
		hanson_hirshman_w7x[i] += hh[i]

		# shift
		l_shift = l_shiftcurvature(p, NS)
		hh_shift[i] = (error_func(biot_savart_hanson_hirshman(r_eval, l_shift)))
		hanson_hirshman_shift_w7x[i] += hh_shift[i]

	axes[1].loglog(NSs, hh, alpha=0.25, color = 'blue', linewidth=0.5)#, label = "Shifted piecewise linear")
	axes[1].loglog(NSs, hh_shift, alpha=0.25, color = 'orange', linewidth=0.5)




##################################################################
# PLOTTING
##################################################################

axes[1].loglog(NSs, hanson_hirshman_w7x / R)#, label = "Standard piecewise linear")
axes[1].scatter(NSs, hanson_hirshman_w7x / R)
axes[1].loglog(NSs, hanson_hirshman_shift_w7x / R)#, label = "Shifted piecewise linear")
axes[1].scatter(NSs, hanson_hirshman_shift_w7x / R)
axes[1].set_title("Non-Planar Coil", fontsize=title_size)
#plt.xlabel("Number of discretization points")
#plt.ylabel("Normalized error in magnetic field")
#plt.xlim((6.5,15000))
#plt.ylim((1e-11,1e-1))
#plt.legend()
axes[1].grid(True, which="major")

##################################################################
# END PLOTTING
##################################################################



##################################################################
# D-shaped Coil
##################################################################



r2 = 2.0
r1 = 0.5
k = .5 * np.log(r2/r1)

# arclength distance from r=1.0 to r=2.0
s_f_outer = 2.4960795690997455
# arclength distance from r=1.0 to r=0.5
s_f_inner = -1.0343434551534787

# accuracy of ODE integration (as high as possible)
atol=1e-10
rtol=1e-7
# Height in z of coil, need to shift by this amount
z_f_outer = 1.208227533517828
z_f_inner = 0.4072928588485294
straight = z_f_outer - z_f_inner

def get_points(NS):
	""" 
	Returns three arrays of NS points along the curved section of the D-shaped coil.

	The first array is shape NS x 3 and returns the points, l. 
	The second array is shape NS x 3 and returns the tangent dl = dl/ds * ds.
	The third array is shape NS and returns the spacing in arclength, ds. 

	2/5ths of the points are along the "inner" section from r=0.5 to r=1.0, and 3/5ths
	of the points are along the "outer" section from r=1.0 to r=2.0. On each section,
	the points are equally spaced in arclength.
 	"""

	def dz_ds(r):
		return np.log(r)

	def dr_ds(r):
		return np.sqrt(k**2 - np.log(r)**2)

	def ode_func(s, y):
		z = dz_ds(y[0])
		r = dr_ds(y[0])
		return np.asarray([r, z])

	y0 = np.asarray([1.0, -z_f_outer])

	assert NS % 20 == 0
	NS_straight = int(4 * NS // 20)
	NS_inner = int(4 * NS // 20)
	NS_outer = int(6 * NS // 20)
	z_straight = np.linspace(1.0 - 1/(NS_straight), -1.0 + 1/NS_straight, NS_straight) * straight
	r_straight = np.ones(NS_straight) * r1
	dz_straight = -np.ones(NS_straight) * (2 * straight / NS_straight)
	dr_straight = np.zeros(NS_straight)

	s_eval_inner = np.linspace(0.0 + 1 / (2 * NS_inner), 1.0 - 1 / (2 * NS_inner), NS_inner) * s_f_inner
	res_inner = solve_ivp(ode_func, t_span=(0.0, s_f_inner), y0=y0, t_eval=s_eval_inner, atol=atol, rtol=rtol)
	r_inner = np.flip(res_inner.y[0,:])
	z_inner = np.flip(res_inner.y[1,:])
	deltas_inner = -s_f_inner / NS_inner
	dr_inner = dr_ds(r_inner) * deltas_inner
	dz_inner = dz_ds(r_inner) * deltas_inner
	ds_inner = np.ones(NS_inner) * s_f_inner / NS_inner

	s_eval_outer = np.linspace(0.0 + 1 / (2 * NS_outer), 1.0 - 1 / (2 * NS_outer), NS_outer) * s_f_outer
	res_outer = solve_ivp(ode_func, t_span=(0.0, s_f_outer), y0=y0, t_eval=s_eval_outer, atol=atol, rtol=rtol)
	r_outer = res_outer.y[0,:]
	z_outer = res_outer.y[1,:]
	deltas_outer = s_f_outer / NS_outer
	dr_outer = dr_ds(r_outer) * deltas_outer
	dz_outer = dz_ds(r_outer) * deltas_outer
	ds_outer = np.ones(NS_outer) * s_f_outer / NS_outer
	
	rs = np.concatenate( (r_inner, r_outer,  np.flip(r_outer),  np.flip(r_inner)) )
	zs = np.concatenate( (z_inner, z_outer, -np.flip(z_outer), -np.flip(z_inner)) )
	ys = np.zeros(rs.shape)
	l = np.concatenate((rs[:,np.newaxis], ys[:, np.newaxis], zs[:,np.newaxis]),axis=1)

	drs = np.concatenate( (dr_inner, dr_outer, -np.flip(dr_outer), -np.flip(dr_inner)) )
	dzs = np.concatenate( (dz_inner, dz_outer,  np.flip(dz_outer),  np.flip(dz_inner)) )
	dys = np.zeros(drs.shape)
	dl = np.concatenate((drs[:,np.newaxis], dys[:, np.newaxis], dzs[:,np.newaxis]),axis=1)

	ds = np.concatenate( (ds_inner, ds_outer,  np.flip(ds_outer),  np.flip(ds_inner)) )

	return l, dl, ds

def get_points_shift(l, ds):
	"""
	Takes two arrays, l and ds, and returns a single array of points shifted
	in the outwards normal direction by an amount kappa L^2/12.
	"""

	def dz_ds(r):
		return np.log(r)

	def dr_ds(r):
		return np.sqrt(k**2 - np.log(r)**2)

	def d2z_ds2(r):
		return np.sqrt(k**2 - np.log(r)**2) / r

	def d2r_ds2(r):
		return - np.log(r) / r

	def r1(r):
		return np.concatenate((dr_ds(r)[:, np.newaxis], np.zeros(r.shape)[:, np.newaxis], dz_ds(r)[:, np.newaxis]), axis=1)

	def r2(r):
		return np.concatenate((d2r_ds2(r)[:, np.newaxis], np.zeros(r.shape)[:, np.newaxis], d2z_ds2(r)[:, np.newaxis]), axis=1)

	def curvature(r):
		one = r1(r)
		two = r2(r)
		return np.linalg.norm(np.cross(one, two), axis=-1) / np.linalg.norm(one, axis=-1)**3

	def tangent(r):
		return r1(r) / np.linalg.norm(r1(r), axis=-1)[:,np.newaxis]

	def normal(r):
		one = r1(r)
		two = r2(r)
		T = tangent(r)
		bottom = np.linalg.norm(one, axis=-1)**2
		top = two - T * (dot(T, two))[:, np.newaxis]
		dTds = top / bottom[:, np.newaxis]
		return dTds / np.linalg.norm(dTds, axis=-1)[:, np.newaxis]
	NS = l.shape[0]
	r = l[: (NS//2), 0]
	z = l[: (NS//2), 2]
	normal = normal(r)
	curvature = curvature(r)
	
	L = np.linalg.norm(r1(r) * ds[: NS // 2, np.newaxis], axis=-1)
	l_shift = l[: NS//2, :] - (curvature * L**2)[:, np.newaxis] * normal / 12
	r_shift = l_shift[:,0]
	z_shift = l_shift[:,2]
	r_new = np.concatenate( (r_shift, np.flip(r_shift)))
	z_new = np.concatenate( (z_shift, -np.flip(z_shift)))
	return np.concatenate((r_new[:, np.newaxis], np.zeros(r_new.shape)[:, np.newaxis], z_new[:, np.newaxis]), axis=1)


def get_points_quad2(NS):
	"""
	Returns two arrays, l and dl. The points l are created for n=2 quadrature.
	"""
	def dz_ds(r):
		return np.log(r)

	def dr_ds(r):
		return np.sqrt(k**2 - np.log(r)**2)

	def ode_func(s, y):
		z = dz_ds(y[0])
		r = dr_ds(y[0])
		return np.asarray([r, z])

	y0 = np.asarray([1.0, -z_f_outer])
	assert NS % 20 == 0
	NS_straight = int(4 * NS // 20)
	NS_inner = int(4 * NS // 20)
	NS_outer = int(6 * NS // 20)
	z_straight = np.linspace(1.0 - 1/(NS_straight), -1.0 + 1/NS_straight, NS_straight)
	z_straight[::2] -= 2 / NS_straight * (1/2 - 1/np.sqrt(3))
	z_straight[1::2] += 2 / NS_straight * (1/2 - 1 / np.sqrt(3))
	z_straight = z_straight * straight
	r_straight = np.ones(NS_straight) * r1
	dz_straight = -np.ones(NS_straight) * (2 * straight / NS_straight)
	dr_straight = np.zeros(NS_straight)

	s_eval_inner = np.linspace(0.0 + 1 / (2 * NS_inner), 1.0 - 1 / (2 * NS_inner), NS_inner)
	s_eval_inner[::2] += 1 / NS_inner * (1/2 - 1/np.sqrt(3))
	s_eval_inner[1::2] -= 1 / NS_inner * (1/2 - 1 / np.sqrt(3))
	s_eval_inner = s_eval_inner * s_f_inner
	res_inner = solve_ivp(ode_func, t_span=(0.0, s_f_inner), y0=y0, t_eval=s_eval_inner, atol=atol, rtol=rtol)
	r_inner = np.flip(res_inner.y[0,:])
	z_inner = np.flip(res_inner.y[1,:])
	deltas_inner = -s_f_inner / NS_inner
	dr_inner = dr_ds(r_inner) * deltas_inner
	dz_inner = dz_ds(r_inner) * deltas_inner

	s_eval_outer = np.linspace(0.0 + 1 / (2 * NS_outer), 1.0 - 1 / (2 * NS_outer), NS_outer)
	s_eval_outer[::2] += 1 / NS_outer * (1/2 - 1/np.sqrt(3))
	s_eval_outer[1::2] -= 1 / NS_outer * (1/2 - 1 / np.sqrt(3))
	s_eval_outer = s_eval_outer * s_f_outer
	res_outer = solve_ivp(ode_func, t_span=(0.0, s_f_outer), y0=y0, t_eval=s_eval_outer, atol=atol, rtol=rtol)
	r_outer = res_outer.y[0,:]
	z_outer = res_outer.y[1,:]
	deltas_outer = s_f_outer / NS_outer
	dr_outer = dr_ds(r_outer) * deltas_outer
	dz_outer = dz_ds(r_outer) * deltas_outer
	rs = np.concatenate( (r_inner, r_outer,  np.flip(r_outer),  np.flip(r_inner)) )
	zs = np.concatenate( (z_inner, z_outer, -np.flip(z_outer), -np.flip(z_inner)) )
	ys = np.zeros(rs.shape)
	l = np.concatenate((rs[:,np.newaxis], ys[:, np.newaxis], zs[:,np.newaxis]),axis=1)

	drs = np.concatenate( (dr_inner, dr_outer, -np.flip(dr_outer), -np.flip(dr_inner)) )
	dzs = np.concatenate( (dz_inner, dz_outer,  np.flip(dz_outer),  np.flip(dz_inner)) )
	dys = np.zeros(drs.shape)
	dl = np.concatenate((drs[:,np.newaxis], dys[:, np.newaxis], dzs[:,np.newaxis]),axis=1)

	return l, dl



def d_shaped_B(NS, r_eval):
	"""
	Computes the magnetic field for n=1 continuous tangent, piecewise linear,
	n=2 continuous tangent, and shifted piecewise linear, at the point r_eval.
	"""
	l, dl, ds = get_points(NS)
	l_end, ds_end = get_points_endpoints(NS)
	l_quad2, dl_quad2 = get_points_quad2(NS)
	l_shift = get_points_shift_endpoints(l_end, ds_end)
	if (NS - l.shape[0] != 0):
		print("Not same")
		print(l.shape[0])
	# Hanson and Hirshman needs the endpoints as well, so I append those.
	# This is necessary because get_points doesn't return the endpoints of the D-shaped straight section.
	l_hh = np.concatenate((np.asarray([r1, 0.0, -straight])[np.newaxis, :], l, np.asarray([r1, 0.0, straight])[np.newaxis, :]),axis=0)
	#l_hh_shift = np.concatenate((np.asarray([r1, 0.0, -straight])[np.newaxis, :], l_shift, np.asarray([r1, 0.0, straight])[np.newaxis, :]),axis=0)
	B_analytic_dl = biot_savart(r_eval, l, dl)
	B_hh = biot_savart_hanson_hirshman(r_eval, l_hh, wrap=False)
	B_hh_shift = biot_savart_hanson_hirshman(r_eval, l_shift, wrap=False)
	B_analytic_quad2 = biot_savart(r_eval, l_quad2, dl_quad2)
	return B_analytic_dl, B_hh, B_analytic_quad2, B_hh_shift


def get_points_endpoints(NS):
	""" 
	Returns three arrays of NS points along the curved section of the D-shaped coil.

	The first array is shape NS x 3 and returns the points, l. 
	The second array is shape NS x 3 and returns the tangent dl = dl/ds * ds.
	The third array is shape NS and returns the spacing in arclength, ds. 

	2/5ths of the points are along the "inner" section from r=0.5 to r=1.0, and 3/5ths
	of the points are along the "outer" section from r=1.0 to r=2.0. On each section,
	the points are equally spaced in arclength.
 	"""

	def dz_ds(r):
		return np.log(r)

	def dr_ds(r):
		return np.sqrt(k**2 - np.log(r)**2)

	def ode_func(s, y):
		z = dz_ds(y[0])
		r = dr_ds(y[0])
		return np.asarray([r, z])

	y0 = np.asarray([1.0, -z_f_outer])

	assert NS % 20 == 0
	NS_inner = int(4 * NS // 20)
	NS_outer = int(6 * NS // 20)

	#############
	# Here is where the non-constant spacing in Appendix D is applied
	ds = 1/(NS_inner - 2 + np.sqrt(2))
	ds_prime = ds/np.sqrt(2)
	new_eval = np.linspace(ds_prime, 1.0 - ds_prime, NS_inner-1)
	s_eval_inner = new_eval * s_f_inner
	#############
	res_inner = solve_ivp(ode_func, t_span=(0.0, s_f_inner), y0=y0, t_eval=s_eval_inner, atol=atol, rtol=rtol)
	r_inner = np.flip(res_inner.y[0,:])
	z_inner = np.flip(res_inner.y[1,:])
	deltas_inner = -s_f_inner / NS_inner
	dr_inner = dr_ds(r_inner) * deltas_inner
	dz_inner = dz_ds(r_inner) * deltas_inner
	ds_inner = np.ones(NS_inner-1) * s_f_inner * ds

	#############
	# Here is where the non-constant spacing in Appendix D is applied
	ds = 1/(NS_outer - 2 + np.sqrt(2))
	ds_prime = ds/np.sqrt(2)
	new_eval = np.linspace(ds_prime, 1.0 - ds_prime, NS_outer-1)
	new_eval = np.concatenate(([0.0], new_eval))
	s_eval_outer = new_eval * s_f_outer
	#############
	#s_eval_outer = np.linspace(0.0, 1.0, NS_outer+1)[:-1] * s_f_outer
	res_outer = solve_ivp(ode_func, t_span=(0.0, s_f_outer), y0=y0, t_eval=s_eval_outer, atol=atol, rtol=rtol)
	r_outer = res_outer.y[0,:]
	z_outer = res_outer.y[1,:]
	deltas_outer = s_f_outer / NS_outer
	dr_outer = dr_ds(r_outer) * deltas_outer
	dz_outer = dz_ds(r_outer) * deltas_outer
	ds_outer = np.ones(NS_outer) * s_f_outer * ds
	ds_outer[0] = 0.0
	
	rs = np.concatenate( (r_inner, r_outer,  np.flip(r_outer),  np.flip(r_inner)) )
	zs = np.concatenate( (z_inner, z_outer, -np.flip(z_outer), -np.flip(z_inner)) )
	ys = np.zeros(rs.shape)
	l = np.concatenate((rs[:,np.newaxis], ys[:, np.newaxis], zs[:,np.newaxis]),axis=1)

	drs = np.concatenate( (dr_inner, dr_outer, -np.flip(dr_outer), -np.flip(dr_inner)) )
	dzs = np.concatenate( (dz_inner, dz_outer,  np.flip(dz_outer),  np.flip(dz_inner)) )
	dys = np.zeros(drs.shape)
	dl = np.concatenate((drs[:,np.newaxis], dys[:, np.newaxis], dzs[:,np.newaxis]),axis=1)

	ds = np.concatenate( (ds_inner, ds_outer,  np.flip(ds_outer),  np.flip(ds_inner)) )

	return l, ds



def get_points_shift_endpoints(l, ds):
	"""
	Takes two arrays, l and ds, and returns a single array of points shifted
	in the outwards normal direction by an amount kappa L^2/12.
	"""

	def dz_ds(r):
		return np.log(r)

	def dr_ds(r):
		return np.sqrt(k**2 - np.log(r)**2)

	def d2z_ds2(r):
		return np.sqrt(k**2 - np.log(r)**2) / r

	def d2r_ds2(r):
		return - np.log(r) / r

	def r1(r):
		return np.concatenate((dr_ds(r)[:, np.newaxis], np.zeros(r.shape)[:, np.newaxis], dz_ds(r)[:, np.newaxis]), axis=1)

	def r2(r):
		return np.concatenate((d2r_ds2(r)[:, np.newaxis], np.zeros(r.shape)[:, np.newaxis], d2z_ds2(r)[:, np.newaxis]), axis=1)

	def curvature(r):
		one = r1(r)
		two = r2(r)
		return np.linalg.norm(np.cross(one, two), axis=-1) / np.linalg.norm(one, axis=-1)**3

	def tangent(r):
		return r1(r) / np.linalg.norm(r1(r), axis=-1)[:,np.newaxis]

	def normal(r):
		one = r1(r)
		two = r2(r)
		T = tangent(r)
		bottom = np.linalg.norm(one, axis=-1)**2
		top = two - T * (dot(T, two))[:, np.newaxis]
		dTds = top / bottom[:, np.newaxis]
		return dTds / np.linalg.norm(dTds, axis=-1)[:, np.newaxis]
	NS = l.shape[0]
	r = l[: (NS//2), 0]
	z = l[: (NS//2), 2]
	normal = normal(r)
	curvature = curvature(r)
	
	L = np.linalg.norm(r1(r) * ds[: NS // 2, np.newaxis], axis=-1)
	l_shift = l[: NS//2, :] - (curvature * L**2)[:, np.newaxis] * normal / 12
	r_shift = l_shift[:,0]
	z_shift = l_shift[:,2]
	r_new = np.concatenate( ([0.5], r_shift, [2.0], np.flip(r_shift), [0.5]))
	z_new = np.concatenate( ([-straight], z_shift, [0.0], -np.flip(z_shift), [straight]))
	return np.concatenate((r_new[:, np.newaxis], np.zeros(r_new.shape)[:, np.newaxis], z_new[:, np.newaxis]), axis=1)


# Tests to plot coil shape to demonstrate it is working correctly
"""
N_plot = 200
font = {'size'   : 14}
plt.rc('font', **font)
l, dl, _ = get_points(N_plot)
l = np.concatenate((l[:], l[0:1]))
plt.plot(l[:,0], l[:,2])
plt.title("D-Shaped Coil")
plt.axis("equal")
plt.xlabel("r")
plt.ylabel("z")
#for i in range(N_plot):
	#plt.arrow(l[i,0] - dl[i,0]/2, l[i,2] - dl[i,2]/2, dl[i,0], dl[i,2],width= 0.0001,length_includes_head=True)
plt.show()

N_plot = 20
l, dl = get_points_quad2(N_plot)
plt.scatter(l[:,0], l[:,2])
plt.axis("equal")
#for i in range(N_plot):
	#plt.arrow(l[i,0] - dl[i,0]/2, l[i,2] - dl[i,2]/2, dl[i,0], dl[i,2],width= 0.0001,length_includes_head=True)
plt.show()

N_plot = 20
l, dl = get_points_new(N_plot)
plt.scatter(l[:,0], l[:,2])
plt.axis("equal")
#for i in range(N_plot):
	#plt.arrow(l[i,0] - dl[i,0]/2, l[i,2] - dl[i,2]/2, dl[i,0], dl[i,2],width= 0.0001,length_includes_head=True)
plt.show()
N_plot = 20
l, dl, ds = get_points(N_plot)
l_shift = get_points_shift(l, ds)
plt.scatter(l[:,0], l[:,2])
plt.scatter(l_shift[:,0], l_shift[:, 2])
plt.axis("equal")
plt.show()
"""

R=1



NSs = np.asarray([20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000, 10000])#6000, 32000, 64000])#, 128000]
quadrature = np.zeros(NSs.shape)
hanson_hirshman_D = np.zeros(NSs.shape)
mixed = np.zeros(NSs.shape)
quadrature_n2 = np.zeros(NSs.shape)
mixed2 = np.zeros(NSs.shape)
hanson_hirshman_shift_D = np.zeros(NSs.shape)

for r in range(R):
	rand_vec = (rand((3)) - 0.5) * 0.4
	r_eval = np.asarray([1.0, 0.0, 0.0]) + rand_vec	
	B_exact_quad, B_exact_hh, B_exact_quad2, B_exact_hh_shift = d_shaped_B(NS_max, r_eval)
	error_func = partial(error_in_B, B_exact_hh)

	hh = np.zeros(NSs.shape)
	hh_shift = np.zeros(NSs.shape)
	for i in range(NSs.shape[0]):
		NS = int(NSs[i])
		print(NS)
		B1, B2, B4, B5 = d_shaped_B(NS, r_eval)
		quadrature[i] += error_func(B1)
		hh[i] = (error_func(B2))
		hanson_hirshman_D[i] += hh[i]
		quadrature_n2[i] += (error_func(B4))
		hh_shift[i] = (error_func(B5))
		hanson_hirshman_shift_D[i] += hh_shift[i]

	axes[2].loglog(NSs, hh, alpha=0.25, color = 'blue', linewidth=0.5)#, label = "Shifted piecewise linear")
	axes[2].loglog(NSs, hh_shift, alpha=0.25, color = 'orange', linewidth=0.5)


##################################################################
# PLOTTING
##################################################################


axes[2].loglog(NSs, hanson_hirshman_D / R)#, label = "Standard piecewise linear")
axes[2].scatter(NSs, hanson_hirshman_D / R)
axes[2].loglog(NSs, hanson_hirshman_shift_D / R)#, label = "Shifted piecewise linear")
axes[2].scatter(NSs, hanson_hirshman_shift_D / R)
axes[2].set_title("D-Shaped Coil", fontsize=title_size)
axes[2].grid(True, which="major")


#axes[2].xlabel("Number of discretization points")
#axes[2].ylabel("Normalized error in magnetic field")
#axes[2].legend()
fig.legend(loc=(0.135,0.79), prop={'size': 11})
fig.tight_layout()

plt.show()

##################################################################
# END PLOTTING
##################################################################
