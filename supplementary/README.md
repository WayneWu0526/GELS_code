# Supplementary source

`Supplementary.tex`, `references.bbl`, and `figures/` form a standalone source bundle for the current three-page document. No manuscript source, main-paper auxiliary file, or private bibliography database is required.

From the repository root:

```bash
./supplementary/build.sh
```

This requires a LaTeX installation containing IEEEtran and the packages declared in the TeX file. It compiles inside `supplementary/.build/` and updates `Supplementary.pdf` at the repository root.

Revision wrappers have been removed. Main-paper references were checked against the current eight-page manuscript: shape metric (15), proposed joint update (16), and experimental Fig. 6. These are plain reference numbers so the PDF does not contain broken links to an absent local `main_RAL.pdf`. Update them if the main-paper numbering changes.

The literature comparison remains as Table II in this source and PDF. It is not a simulation result. The bibliography is included as editable formatted LaTeX in `references.bbl`.

On 29 October 2026 the detailed distance-bound derivation was removed to match the current supplementary document. The bound and its brief derivation source remain in the main manuscript.
