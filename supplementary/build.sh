#!/bin/sh
set -eu
cd "$(dirname "$0")"
latexmk -norc -pdf -interaction=nonstopmode -halt-on-error -outdir=.build Supplementary.tex
cp .build/Supplementary.pdf ../Supplementary.pdf
