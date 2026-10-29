# Verification on 29 October 2026

The distance runner was executed from the revised distribution without any raw localization acquisitions. All 81,804 synthetic trials and 612 summary rows completed with no numerical failures. The trial table, summary table, ray geometry, and both 41-row paper plotting tables agree with the prior release within absolute tolerance 1e-6 and relative tolerance 1e-8. The model, seed, draw order, estimator, source moments, and sample counts are unchanged.

The affine calibration fit was rerun on all 15 CSV files and 3,706 samples. Per-condition and per-sample results agree with the archived tables. The core synthetic example gives 0.0463 mm position error and 0.0534 degrees orientation error.

The numerical code, inputs, and reference results for both source-geometry studies and the motion study are byte-identical to the previously verified release. The only motion change removes the raw-acquisition inventory from a provenance document, which is not read by the numerical entry point. The earlier full runs covered 75,000 continuous-geometry cases, 30,000 five-configuration cases, and 11,520 motion cases. Those unchanged studies were not rerun for this scope change.

The distance plotting command was checked using the newly generated simulation outputs. It creates synthetic distance and noise diagnostics and no longer requires experimental comparison tables.

`verification.json` records the numerical comparisons and preserved-file checks. These checks cover simulation and calibration reproduction. The original localization experiments and the field-to-moment fit are outside this distribution.

## 29 October 2026 publication

The current supplementary document omits the detailed distance-bound derivation. The simulation code and calibration data are unchanged from the verified simulation and calibration distribution.
