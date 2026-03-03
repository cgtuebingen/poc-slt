import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
# import sys
# sys.path.append("../")


def depth_map_plot(depth_map: np.array, resolution: int, title: str):
    min_depth = np.min(depth_map)
    max_depth = np.max(depth_map)
    # print("\n min_depth: ", min_depth, " , max_depth: ", max_depth)
    value_limit = max(np.abs(min_depth), np.abs(max_depth))
    # print("\n value_limit: ", -value_limit, " , value_limit: ", value_limit)

    fig = Figure(figsize=(resolution/10, resolution/20))
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=8)
    ax.set_axis_off()
    plot = ax.imshow(depth_map, origin="lower", cmap="seismic", interpolation="none")
    # plot.set_clim(min_depth, max_depth)
    fig.colorbar(plot, ax=ax)
    fig.tight_layout()
    canvas.draw()
    image = np.array(canvas.renderer.buffer_rgba())

    return image