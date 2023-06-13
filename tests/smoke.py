from dataclasses import dataclass

import jax.numpy as jnp  # noqa: F401
import jupytext
import numpy as np


@dataclass
class Result:
    path: str
    true_mean: float
    true_stddev: float
    true_objective: float

    def __post_init__(self):
        self.name: str = self.path.split("/")[-1].split(".")[0].replace("_", "-")

    def test(
        self,
        objective_variable="history",
        mean_variable="predictive_mean",
        stddev_variable="predictive_std",
    ):
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
        contents = "\n".join([l for l in lines if not l.startswith("%")])

        loc = {}
        exec(contents, globals(), loc)
        try:
            np.testing.assert_almost_equal(
                loc[objective_variable][-1], self.true_objective
            )
            print(f"✅ {self.name} passed objective test")
        except AssertionError:
            print(f"❌ {self.name} failed objective test")
        try:
            np.testing.assert_almost_equal(loc[mean_variable].sum(), self.true_mean)
            print(f"✅ {self.name} passed mean test")
        except AssertionError:
            print(f"❌ {self.name} failed mean test")
        try:
            np.testing.assert_almost_equal(loc[stddev_variable].sum(), self.true_stddev)
            print(f"✅ {self.name} passed stddev test")
        except AssertionError:
            print(f"❌ {self.name} failed stddev test")


regression = Result(
    path="docs/examples/regression.py",
    true_objective=55.07405622,
    true_mean=36.24383416,
    true_stddev=197.04727051,
)
regression.test()

sparse = Result(
    path="docs/examples/collapsed_vi.py",
    true_objective=1924.7634809,
    true_mean=-8.39869652,
    true_stddev=255.74838027,
)
sparse.test()

stochastic = Result(
    path="docs/examples/uncollapsed_vi.py",
    true_objective=-985.71726453,
    true_mean=-54.14787028,
    true_stddev=116.16651265,
)
stochastic.test(mean_variable="meanf", stddev_variable="sigma")
