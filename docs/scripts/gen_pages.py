from pathlib import Path

import mkdocs_gen_files
from mkdocs_gen_files import Nav
from typing import Iterable, Mapping, Union
import dataclasses


class CustomNav(Nav):
    def __init__(self):
        super().__init__()

    @dataclasses.dataclass
    class Item:
        level: int
        title: str
        filename: Union[str, None]

    @classmethod
    def _items(cls, data: Mapping, level: int) -> Iterable[Item]:
        for key, value in data.items():
            if key is not None:
                if key == "gpjax":
                    title = "GPJax"
                elif key == "gps":
                    title = "GPs"
                elif key == "rbf":
                    title = "RBF"
                elif key == "rff":
                    title = "RFF"
                else:
                    title = key.title()

                title = title.replace("_", " ")
                title = title.replace("Matern", "Matérn")

                yield cls.Item(level=level, title=title, filename=value.get(None))
                yield from cls._items(value, level + 1)


nav = CustomNav()

for path in sorted(Path("gpjax").rglob("*.py")):
    module_path = path.relative_to(".").with_suffix("")
    doc_path = path.relative_to("gpjax").with_suffix(".md")
    full_doc_path = Path("api", doc_path)

    parts = list(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
        continue
    elif parts[-1] == "__main__":
        continue

    # final_part = parts[-1].title()
    # parts = parts[:-1] + [final_part]
    nav[parts] = doc_path.as_posix()

    # print(full_doc_path)
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        identifier = ".".join(parts)
        title = identifier.split(".")[-1].replace("_", " ").title()
        if title == "Gps":
            title = "GPs"
        elif title == "Rbf":
            title = "RBF"
        elif title == "Rff":
            title = "RFF"

        if "Matern" in title:
            title = title.replace("Matern", "Matérn")

        # print(f"# {title}\n", file=fd)
        print("::: " + identifier, file=fd)  #

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("api/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
