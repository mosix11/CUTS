import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as patches
import pickle
from PIL import Image

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

def show_figure_grid(figs, rows=None, cols=None, dpi=150, strip_axes=True,
                     labels=None, label_kwargs=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.patches as patches

    if labels is not None and len(labels) != len(figs):
        raise ValueError("labels must have the same length as figs")

    if strip_axes:
        for f in figs:
            for ax in f.axes:
                ax.set_xlabel(''); ax.set_ylabel('')
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values(): s.set_visible(False)

    # rasterize source figs
    imgs = [_figure_to_rgb(f, dpi=dpi) for f in figs]


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

    # default label style
    default_label_kwargs = dict(
        fontsize=10, ha='center', va='top'
    )
    if label_kwargs:
        default_label_kwargs.update(label_kwargs)

    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i < n:
            ax.imshow(imgs[i], aspect='auto')

            # light gray border
            rect = patches.Rectangle((0, 0), 1, 1, linewidth=1,
                                     edgecolor="lightgray", facecolor="none",
                                     transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)

            # bottom label (inside the axes to avoid cropping issues)
            if labels is not None:
                ax.text(0.5, 0.02, labels[i], transform=ax.transAxes,
                        **default_label_kwargs)

    grid_fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.show()
    return grid_fig


def figures_to_gif(figs, out_path, total_duration=6.0, dpi=150,
                   strip_axes=False, background=(255, 255, 255)):
    """
    Save an animated GIF from a list of Matplotlib Figure objects.

    Args:
        figs: list[matplotlib.figure.Figure]
        out_path: str, output GIF path (e.g., 'plots.gif')
        total_duration: float, total length of the GIF in seconds
        dpi: int, rasterization DPI for each figure
        strip_axes: bool, remove ticks/labels/spines before rendering
        background: RGB tuple used for padding when sizes differ
    Returns:
        out_path
    """
    if not figs:
        raise ValueError("No figures provided.")

    if strip_axes:
        for f in figs:
            for ax in f.axes:
                ax.set_xlabel(''); ax.set_ylabel('')
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)

    # Rasterize to numpy arrays
    frames = [_figure_to_rgb(f, dpi=dpi) for f in figs]

    # Pad frames to a common size (GIF requires uniform dimensions)
    h_max = max(fr.shape[0] for fr in frames)
    w_max = max(fr.shape[1] for fr in frames)

    pil_frames = []
    for arr in frames:
        h, w, _ = arr.shape
        base = Image.new("RGB", (w_max, h_max), background)
        im = Image.fromarray(arr)
        # center the frame on the canvas
        x = (w_max - w) // 2
        y = (h_max - h) // 2
        base.paste(im, (x, y))
        pil_frames.append(base)

    # Per-frame duration in ms
    per_frame_ms = int(round(1000 * total_duration / len(pil_frames)))

    # Save GIF
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=per_frame_ms,
        loop=0,          # loop forever
        disposal=2,      # clear between frames
        optimize=True    # smaller file size
    )
    return out_path
    
    
if __name__ == '__main__':
    
    # with open('results/single_experiment/dino_noise_TA/config1/embedding_plots/pca_alpha_4_figs.pkl', 'rb') as f:
    # with open('results/single_experiment/dino_noise_TA/config3/embedding_plots/pca_alpha_4_figs.pkl', 'rb') as f:
    # with open('results/single_experiment/dino_poison_TA/config1/embedding_plots/pca_alpha_4_figs.pkl', 'rb') as f:

    # with open('results/single_experiment/clip_poison_TA/config2/embedding_plots/pca_alpha_4_figs.pkl', 'rb') as f:
    # with open('results/single_experiment/clip_poison_TA/config1/embedding_plots/pca_alpha_4_figs.pkl', 'rb') as f:
    
    # with open('results/single_experiment/clip_noise_TA/config28/embedding_plots/pca_alpha_4_figs.pkl', 'rb') as f:
    # with open('results/single_experiment/clip_noise_TA/config41/embedding_plots/pca_alpha_16_figs.pkl', 'rb') as f:
    # with open('results/single_experiment/clip_noise_TA/config42/embedding_plots/pca_alpha_4_figs.pkl', 'rb') as f:
    # with open('results/single_experiment/clip_noise_TA/config26/embedding_plots/pca_alpha_figs.pkl', 'rb') as f:
    
    # pickle_path =  'results/single_experiment/clip_noise_TA/config28/embedding_plots/pca_alpha_60_figs.pkl'
    # pickle_path =  'results/single_experiment/clip_noise_TA/config41/embedding_plots/pca_alpha_60_figs.pkl'
    # pickle_path =  'results/single_experiment/dino_noise_TA/config1/embedding_plots/pca_alpha_60_figs.pkl'
    # pickle_path =  'results/single_experiment/dino_noise_TA/config3/embedding_plots/pca_alpha_60_figs.pkl'
    
    # pickle_path =  'results/single_experiment/clip_poison_TA/config1/embedding_plots/pca_alpha_60_figs.pkl'
    pickle_path =  'results/single_experiment/clip_poison_TA/config2/embedding_plots/pca_alpha_60_figs.pkl'
    # pickle_path =  'results/single_experiment/dino_poison_TA/config1/embedding_plots/pca_alpha_60_figs.pkl'
    
    with open(pickle_path, 'rb') as f:
    
        figs = pickle.load(f)
    for f in figs:
            plt.close(f)
    # figs = [figs[0], figs[3], figs[5], figs[10]]
    # figs = figs[0:60:5]
    
    # labels = [
    #     r"$\theta_{\text{mix}}$",
    #     r"$\alpha=-0.75$",
    #     r"$\alpha=-1.55$",
    #     r"$\hat{\alpha}^{\ast}_{knn}=-2.3$",
    # ]
    
    # grid_fig = show_figure_grid(figs, rows=1, cols=4)
    
    # grid_fig.savefig("./visulaization_dir/pca_evol_dino_sym_cifar10_40.png", dpi=300, bbox_inches="tight")
    # show_figure_grid(figs)
    
    
    figures_to_gif(
        figs,
        # out_path='./visulaization_dir/pca_evol_gif_clip_noise_config28.gif',
        # out_path='./visulaization_dir/pca_evol_gif_clip_noise_config41.gif',
        # out_path='./visulaization_dir/pca_evol_gif_dino_noise_config1.gif',
        # out_path='./visulaization_dir/pca_evol_gif_dino_noise_config3.gif',
        # out_path='./visulaization_dir/pca_evol_gif_clip_poison_config1.gif',
        out_path='./visulaization_dir/pca_evol_gif_clip_poison_config2.gif',
        # out_path='./visulaization_dir/pca_evol_gif_dino_poison_config1.gif',
        total_duration=10,
        dpi=300,
        strip_axes=False,
    )