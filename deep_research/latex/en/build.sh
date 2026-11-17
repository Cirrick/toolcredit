#!/usr/bin/env bash
# Merge per-section bib files and compile. Usage: bash build.sh
set -e
cd "$(dirname "$0")"
cat bib/*.bib > refs.bib
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || { grep -E "^!" -A3 build.log | head -30; exit 1; }
grep -E "Citation .* undefined|Reference .* undefined" build.log | sort -u | head -20 || true
echo "OK: $(pdfinfo main.pdf 2>/dev/null | grep Pages || echo main.pdf built)"
