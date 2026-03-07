import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure


def plot_v(
    gt_sdf_voxel: np.array,
    number_of_slices: int,
    resolution: int,
    title: str,
    plot_range: list[float],
    cmap: str = "seismic",
):
    # min_sdf = np.min(gt_sdf_voxel)
    # max_sdf = np.max(gt_sdf_voxel)
    # value_limit = max(np.abs(min_sdf), np.abs(max_sdf))

    images = []
    titles = []
    for i in range(0, number_of_slices, 1):
        # res = 128 // 2 = 64
        # dpi = 100
        # figsize_x = 64/10 * dpi = 640
        # figsize_y = 64/20 * dpi = 320
        # fig = Figure(figsize=(resolution/10, resolution/20))

        fig = Figure(figsize=(4.0, 3.2))  # 400x320 px
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        center = int(gt_sdf_voxel.shape[0] / 2)
        if number_of_slices + center < gt_sdf_voxel.shape[0]:
            gt_sdf_voxel_slice = gt_sdf_voxel[int(i + center), :, :]
            title = title + "_" + "_slice" + str(i)

            ax.set_title(title, fontsize=8)
            ax.set_axis_off()
            plot = ax.imshow(
                gt_sdf_voxel_slice, origin="lower", cmap=cmap, interpolation="none"
            )

            plot.set_clim(plot_range[0], plot_range[1])
            fig.colorbar(plot, ax=ax)
            fig.tight_layout()
            canvas.draw()
            image = np.array(canvas.renderer.buffer_rgba())
            images.append(image)
            titles.append(title)
    return images, titles
