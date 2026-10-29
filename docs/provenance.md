# Provenance and packaging

This local repository snapshot was assembled on 28 October 2026 from the working RA-L manuscript and its frozen research code and data.

- The core example comes from the manuscript's existing standalone source-code package.
- Calibration preserves the 15 original CSV files and required JSON configurations byte for byte. Its portable fit had previously been compared with the original calibration functions. `calibration/reference_results/original_code_equivalence.json` preserves that earlier comparison.
- Geometry uses the preserved source-geometry simulation workspace. Its archived run records identify original revision `a8c93d1caea87f959f14d0e7b2702db3a6eff608`. Both the numerical scripts and frozen result tables are included.
- Distance now reads `simulation_parameters.json`. Geometries and full-precision moment constants were extracted once from the preserved private archive. The raw fields, historical localization estimates, and experimental comparison outputs are excluded. The synthetic calculations and parameter origins are retained.
- Motion uses the existing frozen 12-sensor circular-motion simulation package and its independent estimator snapshot.

The five RViz panels were extracted directly from the embedded image streams of the existing source-geometry figure PDF. They provide figure-resolution inputs for the plotting script. They are not new RViz captures.

Packaging changes comprise relative paths, explicit fresh-output checks, documentation, and NumPy scalar compatibility (`float(1x1 array)` changed to `.item()`). The distance helper was trimmed to the utility functions used by the sweep. The motion renderer's unused legacy caption was corrected from 20 to 40 mm. No model, seed, noise level, source moment, fit target, or sampling count was changed.

The standalone supplementary TeX removes revision wrappers and replaces external manuscript cross-reference lookups with verified numbers. Its text and rendered pages are checked against the current working supplementary document.

`file_manifest.json` records file sizes and SHA-256 hashes for the distributed snapshot. The separate preparation archive retains original files and packaging differences. Earlier verification records in `reference_results/` describe historical checks. The verification report in this directory records the current local runs.

On 28 October 2026 the authors authorized publication in `WayneWu0526/GELS_code`. The former anonymous-review notice was replaced with a named-author public-inspection notice, retaining all rights reserved. Before upload, the Fig. 1 caption was corrected to identify 0.70–0.90 m as the displayed height range. The sampled volume and numerical results are unchanged.

## Scope update on 29 October 2026

The authors limited the data distribution to simulation inputs and outputs plus calibration measurements. The complete prior local archive is retained separately. The distance runner no longer reads localization measurements or refits moments. It uses the same full-precision constants and reference geometries. Original simulation tables are preserved and checked against a fresh full run. See `verification.md` for the completed checks and `reproduction_scope.md` for the boundary of the release.

## Current publication

The public repository was recreated on 29 October 2026 with a new history containing the simulation and calibration distribution. No raw localization acquisitions or experimental replay results are included. The supplementary source was synchronized with the current manuscript by removing the detailed distance-bound derivation.
