# Computation of the Biot-Savart line integral
A comparison of different methods for computing the Biot-Savart line integral over a 1D current-carrying curve. This code is for the paper ``Computation of the Biot-Savart line integral" (https://arxiv.org/abs/2105.12522). Here, we include the code used to produce the plots shown in the paper.

`experiments_for_paper.py` computes the error for the circular coil, non-planar stellarator-like coil, and D-shaped tokamak-like coil. 

`nonplanar_integrate_half_of_curve.py` demonstrates that the errors in B decrease for a line segment (or for discontinuous curvature) decrease as third-order if points are spaced equally, and decrease as fourth-order if the prescription in appendix C is followed.

`confirm_taylor_expansion.py` confirms that the analytic Taylor expansion in Appendix A is correct, and that the error from a single filament is fourth-order if measured at the center of the coil. Elsewhere, the second-order errors do not cancel.

`demonstrate_variation_in_epsilon_N.py` is the same as `experiments_for_paper.py`, although it outputs shaded lines to demonstrate the variation in \epsilon_N across different measurement points.

`nonplanar_integrate_half_of_curve.py` is a second test (the D-shaped coil being the first) of the prescription in appendix C for a non-closed curve, to confirm that following the prescription results in fourth-order convergence. 

`variation_in_error_with_radius.py` demonstrates that, in regions closer to the coil, the error increases but the order of convergence remains constant.
