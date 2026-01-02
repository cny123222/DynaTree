#!/bin/bash
# LaTeX Compilation Script for DynaTree Paper

set -e  # Exit on error

echo "=========================================="
echo "  DynaTree Paper Compilation"
echo "=========================================="

cd "$(dirname "$0")"

# Clean old auxiliary files
echo "[1/4] Cleaning old files..."
rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz *.fdb_latexmk *.fls

# First compilation
echo "[2/4] First compilation pass..."
pdflatex -interaction=nonstopmode neurips_2025.tex > /dev/null 2>&1 || {
    echo "⚠️  First pass completed with warnings (this is normal)"
}

# Second compilation (for references)
echo "[3/4] Second compilation pass (resolving references)..."
pdflatex -interaction=nonstopmode neurips_2025.tex > /dev/null 2>&1 || {
    echo "⚠️  Second pass completed with warnings"
}

# Check result
echo "[4/4] Checking output..."
if [ -f neurips_2025.pdf ]; then
    FILE_SIZE=$(ls -lh neurips_2025.pdf | awk '{print $5}')
    PAGE_COUNT=$(pdfinfo neurips_2025.pdf 2>/dev/null | grep "Pages:" | awk '{print $2}')
    echo ""
    echo "=========================================="
    echo "✓ Compilation successful!"
    echo "=========================================="
    echo "  File: neurips_2025.pdf"
    echo "  Size: $FILE_SIZE"
    echo "  Pages: $PAGE_COUNT"
    echo ""
    echo "To view the PDF:"
    echo "  xdg-open neurips_2025.pdf"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ Compilation failed!"
    echo "=========================================="
    echo "Check the log file for errors:"
    echo "  less neurips_2025.log"
    echo ""
    exit 1
fi

