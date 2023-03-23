import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
import matplotlib.collections as coll


def set_font(font_path):
    font = mpl.font_manager.FontEntry(fname=font_path, name="my_font")
    mpl.font_manager.fontManager.ttflist.append(font)

    mpl.rcParams.update(
        {
            "font.family": font.name,
        }
    )

if __name__=='__main__':
    set_font('lato.ttf')
    def plot_grid(ax, xs, ys, width, height, col, ncols, nrows, falpha, ealpha, inner_grid = True):
        if inner_grid:
            pat = []
            for xi in xs:
                for yi in ys:
                    sq = patches.Rectangle(xy = (xi, yi),
                                        width = width,
                                        height = height)
                    pat.append(sq)

            pc = coll.PatchCollection(pat,
                                    facecolors=f'#ffffff{str(int(falpha*100)).zfill(2)}',
                                    edgecolors=f'{col}{str(int(ealpha*100)).zfill(2)}',
                                    linewidth=0.5)
            ax.add_collection(pc)

        orig = (min(xs), min(ys))
        patch = patches.Rectangle(xy = orig,
                                  width = width*ncols,
                                  height = height*nrows,
                                  facecolor=f'{col}{str(int(falpha*100)).zfill(2)}',
                                  edgecolor=f'{col}{str(int(ealpha*100)).zfill(2)}',
                                  linewidth=3.0)
        ax.add_patch(patch)


    fig = plt.figure(tight_layout=True)
    ax = plt.subplot(111, aspect='equal')

    wid = 1
    hei = 1
    nrows = 3
    ncols = 3
    xx = np.arange(0, ncols, wid)
    yy = np.arange(0, nrows, hei)
    offset = 0.2
    face_alpha = 0.1
    edge_alpha = 0.99
    f_alpha_decay = 0.9
    e_alpha_decay = 0.9
    cols = ["#5E97F6", "#30A89C", "#9C26B0"] * 2

    for idx, col in enumerate(cols):
        xs = xx + idx * offset
        ys = yy + idx * offset
        if idx <10:
            inner = True
        else:
            inner = False
        plot_grid(ax, xs, ys, wid, hei, col, ncols, nrows, falpha=face_alpha, ealpha=edge_alpha, inner_grid=inner)
        face_alpha *= f_alpha_decay
        edge_alpha *= e_alpha_decay

    text = ax.text(
        x=1.5,
        y=1.45,
        s="JaxLinOp",
        fontsize=45,
        horizontalalignment="center",
        verticalalignment="center",
    )
    text.set_path_effects(
        [
            path_effects.Stroke(linewidth=3, foreground="white"),
            path_effects.Normal(),
        ]
    )

    ax.set_xlim(-2, ncols+2.3)
    ax.set_ylim(-0.4, nrows+1.3)
    plt.axis('off')
    plt.savefig('logo.png', dpi=450, bbox_inches='tight')
    plt.savefig('logo.pdf')