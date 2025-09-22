import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

import pickle

def _figure_to_rgb(fig, dpi=150):
    """Render a Matplotlib Figure to an RGB numpy array."""
    canvas = FigureCanvasAgg(fig)
    orig_dpi = fig.get_dpi()
    fig.set_dpi(dpi)
    canvas.draw()
    w, h = canvas.get_width_height()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    fig.set_dpi(orig_dpi)  # restore
    return buf[..., :3]    # RGB

def show_figure_grid(figs, rows=None, cols=None, dpi=150, strip_axes=True, close_sources=True):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    def _figure_to_rgb(fig, dpi=150):
        canvas = FigureCanvasAgg(fig)
        orig_dpi = fig.get_dpi()
        fig.set_dpi(dpi)
        canvas.draw()
        w, h = canvas.get_width_height()
        import numpy as np
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
        fig.set_dpi(orig_dpi)
        return buf[..., :3]

    # (same rows/cols logic as before)

    if strip_axes:
        for f in figs:
            for ax in f.axes:
                ax.set_xlabel(''); ax.set_ylabel('')
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values(): s.set_visible(False)

    # Render first
    imgs = [_figure_to_rgb(f, dpi=dpi) for f in figs]

    # Now optionally close sources so they won't pop up
    if close_sources:
        for f in figs:
            plt.close(f)

    # Build the grid figure (same as before)
    n = len(figs)
    if cols is None and rows is None:
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
    elif rows is None:
        rows = int(np.ceil(n / cols))
    elif cols is None:
        cols = int(np.ceil(n / rows))

    grid_fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = np.array(axes).reshape(rows, cols)
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i < n:
            ax.imshow(imgs[i], aspect='auto')

    grid_fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.show()
    
with open('results/single_experiment/clip_noise_TA/config26/embedding_plots/pca_alpha_figs.pkl', 'rb') as f:
    figs = pickle.load(f)
    
show_figure_grid(figs)