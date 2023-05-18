import jupytext
from glob import glob
import mkdocs_gen_files


notebooks = glob("docs/examples/*.py", recursive=True)

code = {}
for notebook in notebooks:
    if "utils" in notebook:
        continue
    ntbk = jupytext.read(notebook)
    code_cells = []
    header = ntbk["cells"][0]["source"].split("\n")[0].replace("#", "").strip()
    for c in ntbk["cells"]:
        if c["cell_type"] == "code":
            code_cells.append(c["source"])
    code[header] = "\n\n".join(code_cells)


with mkdocs_gen_files.open("give_me_the_code.md", "w") as f:
    print("# Give me the code", file=f)
    for k, v in code.items():
        print(f"## {k}", file=f)
        print(file=f)
        print("```python", file=f)
        print(v, file=f)
        print("```", file=f)
        print(file=f)
    mkdocs_gen_files.set_edit_path("give_me_the_code.md", "notebook_converter.py")
