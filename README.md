# Computation of the Biot-Savart line integral
A comparison of different methods for computing the Biot-Savart line integral over a 1D current-carrying curve. This code is for the paper ``Computation of the Biot-Savart line integral" (https://arxiv.org/abs/2105.12522). Here, we include the code used to produce the plots shown in the paper.

`experiments_for_paper.py` computes the error for the circular coil, non-planar stellarator-like coil, and D-shaped tokamak-like coil. 

`nonplanar_integrate_half_of_curve.py` demonstrates that the errors in B decrease for a line segment (or for discontinuous curvature) decrease as third-order if points are spaced equally, and decrease as fourth-order if the prescription in appendix D is followed.
