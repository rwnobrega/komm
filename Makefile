.PHONY: docs test

docs:
	@$(MAKE) -C sphinx all

test:
	@python -m pytest komm/ --doctest-modules
	@python -m pytest tests/
