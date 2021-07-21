import numpy as np
from functools import partial
import matplotlib.pyplot as plt
from biotsavart import *
from numpy.random import rand



##################################################################
# Circular Coil
##################################################################


Rs = np.linspace(0, 1.0, 10, endpoint=False)
NSs = np.asarray([10, 12, 14, 16, 20, 24, 30, 40, 50, 60, 100, 200, 400, 800, 1600, 3200, 6400, 10000])

angle = np.random.random() * np.pi * 2



for r in Rs:


	p = np.asarray([[0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
	max_NS = int(1e7)
	r_eval = r * np.asarray([np.cos(angle), np.sin(angle), 0.0])
	l = fourier_to_real_space(p, max_NS)
	dl = dl_dt(p, max_NS) * (2 * np.pi / max_NS)
	B_analytical = biot_savart(r_eval, l, dl)
	error_func = partial(error_in_B, B_analytical)
	
	hh = np.zeros(NSs.shape[0])
	hh_shift = np.zeros(NSs.shape[0])
	quadrature = np.zeros(NSs.shape)
	quadrature_n2 = np.zeros(NSs.shape)
	hanson_hirshman = np.zeros(NSs.shape)
	hanson_hirshman_shift = np.zeros(NSs.shape)

	for i in range(NSs.shape[0]):
		NS = int(NSs[i])
		print(NS)
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
		hanson_hirshman[i] += hh[i]
		# shift
		l_shift = l_shiftcurvature(p, NS)
		hh_shift[i] = (error_func(biot_savart_hanson_hirshman(r_eval, l_shift)))
		hanson_hirshman_shift[i] += hh_shift[i]
	
	font = {'size' : 12}
	plt.rc('font', **font)
	plt.loglog(NSs, hanson_hirshman, alpha = 1/10 + 9*r/10, color = 'blue')
	plt.scatter(NSs, hanson_hirshman, color = 'blue')
	plt.loglog(NSs, hanson_hirshman_shift, alpha = 1/10 + 9*r/10, color = 'orange')
	plt.scatter(NSs, hanson_hirshman_shift, color='orange')
plt.title("Field Error, Circular Coil")
plt.xlabel("Number of discretization points")
plt.ylabel("Normalized error in magnetic field")
plt.xlim((6.5,15000))
plt.ylim((1e-11,1e-1))
plt.grid(True, which="both")
plt.show()