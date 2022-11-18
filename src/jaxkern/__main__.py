"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """JaxKern."""


if __name__ == "__main__":
    main(prog_name="jaxkern")  # pragma: no cover
