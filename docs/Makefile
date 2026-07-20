# Rebuild note.pdf from note.tex.
# Runs pdflatex twice so cross-references and the bibliography resolve.
# Figure paths in note.tex are relative to this directory (../results/...),
# so pdflatex must run from here.

TEX      = note.tex
PDF      = note.pdf
LATEX    = pdflatex
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

.PHONY: all pdf clean

all: pdf

pdf: $(PDF)

$(PDF): $(TEX)
	$(LATEX) $(LATEXFLAGS) $(TEX)
	$(LATEX) $(LATEXFLAGS) $(TEX)

clean:
	rm -f note.aux note.log note.out note.toc
