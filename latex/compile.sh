#!/bin/bash

# This script compiles the LaTeX paper with bibliography.

# Change to the script's directory to ensure paths are correct.
cd "$(dirname "$0")"

FILE="paper"

echo "--- Starting LaTeX Compilation for $FILE.tex ---"

# 1. First pdflatex run to generate .aux file with citation keys
pdflatex "$FILE.tex"

# 2. Run bibtex to process the bibliography
bibtex "$FILE"

# 3. Second pdflatex run to include the bibliography in the document
pdflatex "$FILE.tex"

# 4. Third pdflatex run to ensure all cross-references and citations are correct
pdflatex "$FILE.tex"

echo "--- Compilation Complete ---"
echo "Output file: $FILE.pdf"
