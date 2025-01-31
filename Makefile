ifeq ($(OS),Windows_NT)
	$(error This Makefile is not supported on Windows)
endif

.PHONY: benchmark clean clean-all copy-benchmark-data extra format package-publication package-sponsor plots

paper: paper.tex format plots extra
	@echo "Generating paper!"
	@mkdir -p paper-output
	@pdflatex --file-line-error --interaction=nonstopmode --output-dir=paper-output/ paper.tex
	@bibtex paper-output/paper.aux
	@pdflatex --file-line-error --interaction=nonstopmode --output-dir=paper-output/ paper.tex
	@pdflatex --file-line-error --interaction=nonstopmode --output-dir=paper-output/ paper.tex

format: paper.tex
	@tex-fmt --config style.toml paper.tex

PLOT_PATH = "paper-dependencies/latex-dependencies/plots"
extra:
	@echo "Minifying plots..."
	@echo "Trimming starting..."
	echo "$PLOT_PATH"
	du -h "$(PLOT_PATH)"
	@find $(PLOT_PATH) -type f -iname "*.pdf" -print0 | xargs -0 -I {} pdfcrop {} {}
	du -h "$(PLOT_PATH)"
	

plots:
	@cd source-code/data-visualization && make

clean:
	@echo "Cleaning up!"
	find . -name '.DS_Store' -type f -delete
	rm -rf "paper-output/paper."{abs,aux,bbl,blg,fdb_latexmk,fls,log,out,spl,synctex.gz} \
		   "proof/.lake" ".history"
	@cd source-code/libraries/python && make clean
	@cd source-code/data-visualization && make clean
	
clean-all: clean
	@echo "Cleaning up extra stuff!""
	rm -rf paper-output
	rm -rf source-code/libraries/rust/target

benchmark:
	@echo "Running benchmarks!"
	@echo "Starting Python benchmarks. (ETA: a few hours at most)"
	@echo "Ensure you've run make install-deps in source-code/libraries/python"
	@cd source-code/libraries/python && make benchmark > benchmark-data.txt

	@echo "Starting Rust benchmarks. (ETA: at least a day)"
	@cd source-code/libraries/rust && cargo bench

copy-benchmark-data:
	@cp source-code/libraries/python/benchmark-data.txt paper-dependencies/data-visualization-dependencies/python-benchmark-data.txt
	@cp -r source-code/libraries/rust/target/bench/ paper-dependencies/data-visualization-dependencies/rust-bench

package-publication: clean
	@echo "Creating package.zip"
	rm -rf project project.zip
	zip -r project.zip . -x "*.git*" "*.DS_Store" "paper-output/"*.{abs,aux,bbl,blg,fdb_latexmk,fls,log,out,spl,synctex.gz} \
		"proof/.lake/*" ".history/*" "project.zip" 
