.PHONY: benchmark clean install-deps test

install-deps:
	pip install -r requirements.txt

clean:
	rm -rf .benchmarks .pytest_cache .ruff_cache __pycache__ benchmark-data.txt

test:
	@pytest test_fast_tlsh.py --benchmark-skip

benchmark:
	@pytest test_fast_tlsh.py --benchmark-only
