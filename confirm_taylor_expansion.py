import numpy as np
import matplotlib.pyplot as plt

from biotsavart import biot_savart_hanson_hirshman, error_in_B

# biot_savart_hanson_hirshman(r_eval, l, I = 1.0, wrap=True)
# error_in_B(B_1, B_2)

r_eval = np.asarray([0.0, -1.0, 0.0])

## line segment

N_int = int(1e5)

Ls = [0.05, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]

def r(xs, alpha):
	r = np.zeros((N_int, 3))
	r[:, 0] += xs
	r[:, 1] += alpha
	return r

def r_curve(xs, alpha, kappa, L):
	r = np.zeros((N_int, 3))
	r[:, 0] += xs
	r[:, 1] += L**2 * kappa / 8 - xs**2 * kappa / 2
	return r

errors = np.zeros(len(Ls))

for i, L in enumerate(Ls):
	kappa = 1.0
	alpha = kappa * L**2 / 12

	xs = np.linspace(-L/2 * (1 + alpha * kappa), L/2 * (1 + alpha * kappa), N_int)
	#xs = np.linspace(-L/2, L/2, N_int)
	rs = r(xs, alpha)
	B_straight = biot_savart_hanson_hirshman(r_eval, rs, wrap=False)

	xs = np.linspace(-L/2, L/2, N_int)
	rs_curved = r_curve(xs, alpha, kappa, L)
	B_curved = biot_savart_hanson_hirshman(r_eval, rs_curved, wrap=False)

	errors[i] = error_in_B(B_straight, B_curved)

#plt.scatter(rs[::100, 0], rs[::100, 1])
#plt.scatter(rs_curved[::100, 0], rs_curved[::100, 1])
#plt.show()

plt.loglog(Ls, errors)
plt.scatter(Ls, errors)
plt.show()

