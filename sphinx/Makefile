SPHINXOPTS    = -b "dirhtml" -E -a
SOURCEDIR     = .
BUILDDIR      = _build

all:
	@sphinx-build "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
	@echo "Removing komm.*.rst..."
	@rm -rf komm.*.rst
	@echo "Fixing annoying anchor links..."
	@perl -p -i -e "s/href=\"(komm\..+)#(.+?)\"/href=\"\1\"/" `find _build/ -name *.html`
	@echo "Removing doctrees..."
	@rm -rf _build/doctrees/
	@echo "Moving files to docs..."
	@rm -rf ../docs/docs
	@mkdir -p ../docs/docs
	@mv _build/* ../docs/docs/
	@rm -rf _build/
