# LATEX: pdflatex | xelatex | lualatex
LATEX = lualatex
LATEX_FLAGS = -shell-escape
BIBTEX = bibtex

FILES = $(patsubst %.tex, %.pdf, $(wildcard *.tex))

all: $(FILES)


%.pdf: %.tex
	$(LATEX) $(LATEX_FLAGS) $<
	$(LATEX) $(LATEX_FLAGS) $<
	-$(BIBTEX) `basename $< .tex`
	$(LATEX) $(LATEX_FLAGS) $<
	$(LATEX) $(LATEX_FLAGS) $<
	$(LATEX) $(LATEX_FLAGS) $<
	qpdf --linearize --newline-before-endstream $@ /tmp/$@
	mv /tmp/$@ $@

clean:
	# -rm -f *.{log,toc,tac,aux,dvi,ps,bbl,blg,tmp,nav,out,snm,vrb,rel,cut,abs,xmpi,xmpdata}
	# -rm -rf out _minted* auto
	# for i in $(ALLSUBDIRS); do \
	#     (cd $$i; make clean) || exit 1; \
	# done
	# -rm $(FILES)
	rm -rf *.log *.toc *.out *.aux *.bbl *.blg *.bcf *.run.xml *.synctex.gz *.lof *.lot *.lol *.listing *.nav *.snm *.vrb *.fls *.fdb_latexmk *.xdv *.dvi *.ps *.bak *.rel *.cut *.abs *.xmpi *.xmpdata *.tmp *.tac *.vrb *.synctex.gz
	-rm -rf _minted* auto out

cleanall: clean
	-rm -rf out auto