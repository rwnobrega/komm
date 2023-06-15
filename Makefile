.PHONY: test

test:
	@pytest komm/ --doctest-modules
	@pytest tests/
