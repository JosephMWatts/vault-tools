#!/bin/bash

INPUT_DIR=~/joseph_vault/CPMAI/Source\ Material

for pdf in "$INPUT_DIR"/*.pdf; do
    md="${pdf%.pdf}.md"
    if [ ! -f "$md" ]; then
        pdftotext "$pdf" "$md"
        echo "Converted: $(basename $pdf)"
    else
        echo "Skipped (already exists): $(basename $pdf)"
    fi
done

echo "Done."
