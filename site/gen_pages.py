import inspect
import sys
from pathlib import Path

import mkdocs_gen_files
import yaml
from jinja2 import Environment, FileSystemLoader


def get_data(toc):
    sys.path.append(".")
    import komm

    src_dir = Path(komm.__file__).parents[1]
    repo_url = mkdocs_gen_files.FilesEditor.current().config.repo_url
    src_url = f"{repo_url}blob/v{komm.__version__}/src"

    def _get_source_url(obj):
        lines, start = inspect.getsourcelines(obj)
        path = Path(inspect.getsourcefile(obj)).relative_to(src_dir)
        return f"{src_url}/{path.as_posix()}#L{start}-L{start + len(lines) - 1}"

    def _get_object_data(obj):
        try:
            summary = obj.__doc__.split(".")[0].strip() + "."
        except AttributeError:
            summary = "*No summary available.*"
        return {
            "name": obj.__name__,
            "summary": summary,
            "qualname": f"{obj.__module__}.{obj.__qualname__}",
            "source_url": _get_source_url(obj),
            "options": obj._doc_options() if hasattr(obj, "_doc_options") else None,
        }

    def _get_objects_data(objects):
        return [_get_object_data(komm.__dict__[obj_name]) for obj_name in objects]

    data = {}
    for module, element in toc.items():
        if isinstance(element, list):  # module has no submodules
            data[module] = _get_objects_data(element)
        else:  # module has submodules
            data[module] = {
                submodule: _get_objects_data(objects)
                for submodule, objects in element.items()
            }
    return data


def gen_ref_objects(toc):
    template = JINJA_ENV.get_template("templates/ref/object.md.j2")

    def _gen_ref_file(obj):
        with mkdocs_gen_files.open(f"ref/{obj['name']}.md", "w") as f:
            template.stream(obj=obj).dump(f)

    for elements in toc.values():
        if isinstance(elements, list):  # module has no submodules
            for obj in elements:
                _gen_ref_file(obj)
        else:  # module has submodules
            for objects in elements.values():
                for obj in objects:
                    _gen_ref_file(obj)


def gen_ref_index(data):
    template = JINJA_ENV.get_template("templates/ref/index.md.j2")
    with mkdocs_gen_files.open("ref/index.md", "w") as f:
        template.stream(data=data).dump(f)


def gen_nav(data):
    template = JINJA_ENV.get_template("templates/nav.md.j2")
    with mkdocs_gen_files.open("nav.md", "w") as f:
        template.stream(data=data).dump(f)


JINJA_ENV = Environment(
    loader=FileSystemLoader("site"),
    trim_blocks=True,
    lstrip_blocks=True,
)
toc = yaml.safe_load(open("site/toc.yaml", "r"))
data = get_data(toc)
gen_ref_objects(data)
gen_ref_index(data)
gen_nav(data)
