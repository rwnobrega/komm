.PHONY: test

test:
	@python -m pytest komm/ --doctest-modules
	@python -m pytest tests/
