from dataclasses import (
    dataclass,
    field,
)

from beartype.typing import (
    Any,
    Callable,
    Dict,
)
import jax.numpy as jnp  # noqa: F401
import jupytext

get_last = lambda x: x[-1]


@dataclass
class Result:
    path: str
    comparisons: field(default_factory=dict)
    precision: int = 5

    def __post_init__(self):
        self.name: str = self.path.split("/")[-1].split(".")[0].replace("_", "-")

    def _compare(
        self,
        observed_variables: Dict[str, Any],
        variable_name: str,
        true_value: float,
        operation: Callable[[Any], Any],
    ):
        try:
            value = operation(observed_variables[variable_name])
            assert true_value == value
        except AssertionError as e:
            print(e)

    def test(self):
        notebook = jupytext.read(self.path)
        contents = ""
        for c in notebook["cells"]:
            if c["cell_type"] == "code":
                if c["source"].startswith("%"):
                    pass
                else:
                    contents += c["source"]
            contents += "\n"

        contents = contents.replace('plt.style.use("./gpjax.mplstyle")', "").replace(
            "plt.show()", ""
        )
        lines = contents.split("\n")
        contents = "\n".join([line for line in lines if not line.startswith("%")])

        loc = {}
        exec(contents, globals(), loc)
        for k, v in self.comparisons.items():
            truth, op = v
            self._compare(
                observed_variables=loc, variable_name=k, true_value=truth, operation=op
            )


regression = Result(
    path="docs/examples/regression.py",
    comparisons={
        "history": (55.07405622, get_last),
        "predictive_mean": (36.24383416, jnp.sum),
        "predictive_std": (197.04727051, jnp.sum),
    },
)
regression.test()

sparse = Result(
    path="docs/examples/collapsed_vi.py",
    comparisons={
        "history": (1924.7634809, get_last),
        "predictive_mean": (-8.39869652, jnp.sum),
        "predictive_std": (255.74838027, jnp.sum),
    },
)
sparse.test()

stochastic = Result(
    path="docs/examples/uncollapsed_vi.py",
    comparisons={
        "history": (-985.71726453, get_last),
        "meanf": (-54.14787028, jnp.sum),
        "sigma": (116.16651265, jnp.sum),
    },
)
stochastic.test()
