# Reproduction scope

The public distribution includes the core synthetic example, all source-geometry and distance simulation cases, the circular-motion timing simulation, and the 15 calibration acquisitions with the affine calibration fit. The existing supplementary document remains available.

Raw localization acquisitions, their stored pose estimates, experimental replay outputs, and measured/simulated comparison tables are outside this distribution. The localization experiments in Figs. 6–8 are not reproduced by this package.

Distance simulations use frozen geometric inputs and previously fitted magnetic moments. Their numerical reproduction starts from these constants. The original field-to-moment fit cannot be rerun from this distribution. Source and array geometry retain their origin in tracked measurements and are not presented as independently generated synthetic inputs.

The supplied simulations use the eigenvalue implementation in `compute_rho_hat_eigen`.

Only calibration data are supplied as raw recorded magnetic measurements.
