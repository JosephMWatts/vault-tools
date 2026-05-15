# vault-tools

Utility scripts for managing a local Obsidian knowledge base (second brain).

## convert-pdfs.sh

Converts all PDF files in a target folder to plain-text Markdown (.md).
Skips files that have already been converted — safe to run repeatedly.

Built to ingest CPMAI study materials and other reference PDFs into a 
local, AI-queryable knowledge base powered by Obsidian and Claude Code CLI.

### Usage

```bash
./convert-pdfs.sh
```

### Requirements
- macOS with Homebrew
- pdftotext (install via `brew install poppler`)# vault-tools
