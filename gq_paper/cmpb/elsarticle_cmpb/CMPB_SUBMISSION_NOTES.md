# CMPB LaTeX Submission Notes

Main manuscript:

- `main.tex`
- `references.bib`
- figures: `scribble.png`, `framework.png`, `compare.png`
- separate highlights file: `highlights.tex`

Compile from this directory:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

If `elsarticle.cls` is missing, generate it from the Elsevier bundle first:

```bash
latex elsarticle.ins
```

This local machine currently does not have a TeX executable installed, so PDF compilation was not run here.

Before submission, replace the placeholder author names, affiliations, email, CRediT statement, funding statement, and data/code availability statement with the final information.
