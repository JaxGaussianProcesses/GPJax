from pathlib import Path
import subprocess

EXECUTE = False
EXCLUDE = ["docs/examples/utils.py"]
ALLOW_ERRORS = False


for file in Path("docs/").glob("examples/*.py"):
    if file.as_posix() in EXCLUDE:
        continue

    out_file = file.with_suffix(".md")

    command = "jupytext --to markdown "
    command += f"{'--execute ' if EXECUTE else ''}"
    command += f"{'--allow-errors ' if ALLOW_ERRORS else ''}"
    command += f"{file} --output {out_file}"

    subprocess.run(command, shell=True, check=False)
