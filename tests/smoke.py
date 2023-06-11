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

    def test(self):
        notebook = jupytext.read(self.path)
        contents = ""
        for c in notebook["cells"]:
            if c["cell_type"] == "code":
                if c["source"].startswith("%reload"):
                    pass
                else:
                    contents += c["source"]
            contents += "\n"

        contents = contents.replace('plt.style.use("./gpjax.mplstyle")', "")
        loc = {}
        exec(contents, globals(), loc)
        np.testing.assert_almost_equal(loc["history"][-1], self.true_objective)
        np.testing.assert_almost_equal(loc["predictive_mean"].sum(), self.true_mean)
        np.testing.assert_almost_equal(loc["predictive_std"].sum(), self.true_stddev)


regression = Result(
    path="docs/examples/regression.py",
    true_objective=55.07405622,
    true_mean=36.24383416,
    true_stddev=197.04727051,
)
regression.test()
