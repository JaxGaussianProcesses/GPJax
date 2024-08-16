""" Convert python files in "examples" directory to markdown files using jupytext and nbconvert.

There's only a minor inconvenience with how supporting files are handled by nbconvert,
see https://github.com/jupyter/nbconvert/issues/1164. But these will be under a private
directory `_examples` in the docs folder, so it's not a big deal.

"""
from argparse import ArgumentParser
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil

EXCLUDE = ["utils.py"]


def process_file(file: Path, out_file: Path | None = None, execute: bool = False):
    """Converts a python file to markdown using jupytext and nbconvert."""

    out_dir = out_file.parent
    command = f"cd {out_dir.as_posix()} && "

    out_file = out_file.relative_to(out_dir).as_posix()

    if execute:
        command += f"jupytext --to ipynb {file} --output - "
        command += (
            f"| jupyter nbconvert --to markdown --execute --stdin --output {out_file}"
        )
    else:
        command += f"jupytext --to markdown {file} --output {out_file}"

    subprocess.run(command, shell=True, check=False)


def is_modified(file: Path, out_file: Path):
    """Check if the output file is older than the input file."""
    return out_file.exists() and out_file.stat().st_mtime < file.stat().st_mtime


def main(args):
    # project root directory
    wdir = Path(__file__).parents[2]

    # output directory
    out_dir: Path = args.outdir
    out_dir.mkdir(exist_ok=True, parents=True)

    # copy directories in "examples" to output directory
    for dir in wdir.glob("examples/*"):
        if dir.is_dir():
            (out_dir / dir.name).mkdir(exist_ok=True, parents=True)
            for file in dir.glob("*"):
                # copy, not move!
                shutil.copy(file, out_dir / dir.name / file.name)

    # list of files to be processed
    files = [f for f in wdir.glob("examples/*.py") if f.name not in EXCLUDE]

    # process only modified files
    if args.only_modified:
        files = [f for f in files if is_modified(f, out_dir / f"{f.stem}.md")]

    print(files)

    # process files in parallel
    if args.parallel:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = []
            for file in files:
                out_file = out_dir / f"{file.stem}.md"
                futures.append(
                    executor.submit(
                        process_file, file, out_file=out_file, execute=args.execute
                    )
                )

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error processing file: {e}")
    else:
        for file in files:
            out_file = out_dir / f"{file.stem}.md"
            process_file(file, out_file=out_file, execute=args.execute)


if __name__ == "__main__":
    project_root = Path(__file__).parents[2]

    parser = ArgumentParser()
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--only_modified", action="store_true")
    parser.add_argument(
        "--outdir", type=Path, default=project_root / "docs" / "_examples"
    )
    parser.add_argument("--parallel", type=bool, default=False)
    args = parser.parse_args()

    main(args)
