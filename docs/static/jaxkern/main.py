# import jax.numpy as jnp
# import jax.random as jr
# import matplotlib as mpl
# import matplotlib.patheffects as path_effects
# import matplotlib.pyplot as plt

# import gpjax.kernels as jk

# key = jr.key(123)


# def set_font(font_path):
#     font = mpl.font_manager.FontEntry(fname=font_path, name="my_font")
#     mpl.font_manager.fontManager.ttflist.append(font)

#     mpl.rcParams.update(
#         {
#             "font.family": font.name,
#         }
#     )


# if __name__ == "__main__":
#     set_font("lato.ttf")
#     x1 = jnp.linspace(-3.0, 3.0, 500).reshape(-1, 1)

#     kern = jk.Matern52()
#     focal_points = [-2.5, 0.0, 2.5]
#     cols = ["#5E97F6", "#30A89C", "#9C26B0"]

#     fig, ax = plt.subplots(figsize=(6, 2.5), tight_layout=True)
#     for c, f in zip(cols, focal_points):
#         x2 = jnp.array([[0]])
#         params = kern.init_params(key)
#         Kxx = kern.cross_covariance(params, x1, x2)
#         ax.plot(x1 + f, Kxx, color=c)
#         ax.fill_between((x1 + f).squeeze(), Kxx.squeeze(), color=c, alpha=0.2)
#     ax.axis("off")
#     text = ax.text(
#         x=0.0,
#         y=0.25,
#         s="JaxKern",
#         fontsize=42,
#         horizontalalignment="center",
#         verticalalignment="center",
#     )
#     text.set_path_effects(
#         [
#             path_effects.Stroke(linewidth=3, foreground="white"),
#             path_effects.Normal(),
#         ]
#     )
#     plt.savefig(
#         "logo.png",
#         dpi=450,
#         transparent=True,
#         bbox_inches="tight",
#         pad_inches=0.0,
#     )
