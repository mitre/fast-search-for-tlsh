.PHONY: benchmark clean clean-all generate install-deps test

generate: install-deps
	@for plotter in $(shell ls plot_*.py); do \
		python $$plotter; \
	done

install-deps:
	pip install -r requirements.txt

clean:
	@rm -rf .pytest_cache .ruff_cache __pycache__

clean-all: clean
	@rm -rf "../../paper-dependencies/latex-dependencies/plots/"*

test:
	@pytest test_fast_tlsh.py --benchmark-skip

benchmark:
	@pytest test_fast_tlsh.py --benchmark-only
