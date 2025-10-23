import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.patches as patches
import pickle
from PIL import Image, ImageDraw, ImageFont
from typing import List, Optional, Sequence, Tuple, Union

import os

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
    
    


def figures_to_frames(
    figs_a: List,                         # list[matplotlib.figure.Figure]
    out_dir: Union[str, os.PathLike],
    figs_b: Optional[List] = None,        # optional second list; must match len(figs_a) if provided
    dpi: int = 150,
    strip_axes: bool = False,
    background: Tuple[int, int, int] = (255, 255, 255),

    # Filenames
    prefix: str = "frame",
    start_index: int = 0,
    zero_pad: Optional[int] = None,

    # Layout / styling
    border_px: int = 2,
    border_color: Tuple[int, int, int] = (180, 180, 180),
    gutter_px: int = 0,                   # spacing between left/right panels
    close_figs: bool = False,

    # Optional per-panel numbers (drawn bottom-left inside each bordered panel)
    nums_a: Optional[Sequence[Union[int, float, str]]] = None,
    nums_b: Optional[Sequence[Union[int, float, str]]] = None,
    number_fmt: str = "{:g}",             # used if nums_* items are int/float
    text_color: Tuple[int, int, int] = (0, 0, 0),
    text_margin: Tuple[int, int] = (6, 4),# (x_margin, y_margin) from bottom-left
    font_path: Optional[str] = None,      # custom TTF path
    font_size: Optional[int] = None,      # if None, auto based on panel height
) -> List[str]:
    """
    Render 1 or 2 lists of Matplotlib figures as PNG frames.
    If `figs_b` is provided, each output frame contains two panels (A|B) aligned side-by-side.

    Borders are applied *per panel*. Optional numbers (nums_a / nums_b) are drawn
    bottom-left inside the border for each corresponding panel.

    Returns:
        List[str]: absolute paths to saved frames in order.
    """
    if not figs_a:
        raise ValueError("figs_a is empty.")

    if figs_b is not None and len(figs_b) != len(figs_a):
        raise ValueError("figs_b must have the same length as figs_a.")

    N = len(figs_a)
    os.makedirs(out_dir, exist_ok=True)

    # Strip axes if requested
    if strip_axes:
        for f in figs_a:
            for ax in f.axes:
                ax.set_xlabel(''); ax.set_ylabel('')
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
        if figs_b is not None:
            for f in figs_b:
                for ax in f.axes:
                    ax.set_xlabel(''); ax.set_ylabel('')
                    ax.set_xticks([]); ax.set_yticks([])
                    for s in ax.spines.values():
                        s.set_visible(False)

    # Rasterize all figures to numpy arrays
    frames_a = [_figure_to_rgb(f, dpi=dpi) for f in figs_a]
    frames_b = [_figure_to_rgb(f, dpi=dpi) for f in figs_b] if figs_b is not None else None

    # Compute common panel size across all panels to ensure uniform dims
    h_max = max([fa.shape[0] for fa in frames_a] + ([fb.shape[0] for fb in frames_b] if frames_b else []))
    w_max = max([fa.shape[1] for fa in frames_a] + ([fb.shape[1] for fb in frames_b] if frames_b else []))

    # Prepare font
    # If font_size not given, pick a size that scales with panel height
    # (roughly 5% of panel height but at least 10px)
    auto_font_size = max(10, int(0.05 * h_max))
    fs = font_size or auto_font_size
    try:
        font = ImageFont.truetype(font_path, fs) if font_path else ImageFont.load_default()
        # If default bitmap font looks tiny, try to fall back to truetype if provided
        if font_path is None and isinstance(font, ImageFont.ImageFont):
            # Keep default font; it's small but always available
            pass
    except Exception:
        font = ImageFont.load_default()

    # Determine filename zero-padding
    if zero_pad is None:
        zero_pad = max(2, len(str(start_index + N - 1)))

    # Helper: pad a panel to (w_max, h_max), add border, draw number
    def make_panel(arr, number_text: Optional[str]) -> Image.Image:
        h, w, _ = arr.shape
        base = Image.new("RGB", (w_max, h_max), background)
        im = Image.fromarray(arr)

        # Center the frame on the base
        x = (w_max - w) // 2
        y = (h_max - h) // 2
        base.paste(im, (x, y))

        # Draw border (inside the panel area) as a rectangle around full base
        draw = ImageDraw.Draw(base)
        # Border rectangle around the entire base image
        for k in range(border_px):
            draw.rectangle([k, k, w_max - 1 - k, h_max - 1 - k], outline=border_color)

        # Number (if provided), bottom-left inside the border
        if number_text is not None:
            tx_margin, ty_margin = text_margin
            # Slightly above the bottom edge accounting for border
            text_x = border_px + tx_margin
            text_y = h_max - border_px - ty_margin - fs  # approx baseline
            # Draw text background for readability (optional; commented out)
            # tw, th = draw.textsize(number_text, font=font)
            # draw.rectangle([text_x-2, text_y-2, text_x+tw+2, text_y+th+2], fill=(255,255,255))
            draw.text((text_x, text_y), number_text, fill=text_color, font=font)

        return base

    saved_paths = []
    for i in range(N):
        # Format numbers if provided
        txt_a = None
        if nums_a is not None:
            val = nums_a[i]
            txt_a = number_fmt.format(val) if isinstance(val, (int, float)) else str(val)

        panel_a = make_panel(frames_a[i], txt_a)

        if frames_b is not None:
            txt_b = None
            if nums_b is not None:
                valb = nums_b[i]
                txt_b = number_fmt.format(valb) if isinstance(valb, (int, float)) else str(valb)
            panel_b = make_panel(frames_b[i], txt_b)

            # Combine side-by-side with gutter
            W = w_max * 2 + gutter_px
            H = h_max
            combined = Image.new("RGB", (W, H), background)
            combined.paste(panel_a, (0, 0))
            combined.paste(panel_b, (w_max + gutter_px, 0))
            final_img = combined
        else:
            final_img = panel_a

        # Save
        idx_str = str(start_index + i).zfill(zero_pad)
        fname = f"{prefix}{idx_str}.png"
        fpath = os.path.abspath(os.path.join(out_dir, fname))
        final_img.save(fpath, format="PNG", optimize=True)
        saved_paths.append(fpath)

        if close_figs:
            try:
                import matplotlib.pyplot as plt
                plt.close(figs_a[i])
                if figs_b is not None:
                    plt.close(figs_b[i])
            except Exception:
                pass

    return saved_paths
    
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
    
    pickle_path =  'results/single_experiment/clip_noise_TA/config28/embedding_plots/pca_alpha_60_figs.pkl'
    # pickle_path =  'results/single_experiment/clip_noise_TA/config41/embedding_plots/pca_alpha_60_figs.pkl'
    # pickle_path =  'results/single_experiment/dino_noise_TA/config1/embedding_plots/pca_alpha_60_figs.pkl'
    # pickle_path =  'results/single_experiment/dino_noise_TA/config3/embedding_plots/pca_alpha_60_figs.pkl'
    
    # pickle_path =  'results/single_experiment/clip_poison_TA/config1/embedding_plots/pca_alpha_60_figs.pkl'
    # pickle_path =  'results/single_experiment/clip_poison_TA/config2/embedding_plots/pca_alpha_60_figs.pkl'
    # pickle_path =  'results/single_experiment/dino_poison_TA/config1/embedding_plots/pca_alpha_60_figs_centroids.pkl'
    # pickle_path =  'results/single_experiment/dino_poison_TA/config1/embedding_plots/pca_alpha_60_figs_points.pkl'
    
    with open(pickle_path, 'rb') as f:
        figs = pickle.load(f)
    for f in figs:
        plt.close(f)
        
    # figs = [figs[0], figs[3], figs[5], figs[10]]
    figs = figs[0:60:2]
    
    # labels = [
    #     r"$\theta_{\text{mix}}$",
    #     r"$\alpha=-0.75$",
    #     r"$\alpha=-1.55$",
    #     r"$\hat{\alpha}^{\ast}_{knn}=-2.3$",
    # ]
    
    # grid_fig = show_figure_grid(figs, rows=1, cols=4)
    
    # grid_fig.savefig("./visulaization_dir/pca_evol_dino_sym_cifar10_40.png", dpi=300, bbox_inches="tight")
    # show_figure_grid(figs)
    
    
    # figures_to_gif(
    #     figs,
    #     # out_path='./visulaization_dir/pca_evol_gif_clip_noise_config28.gif',
    #     # out_path='./visulaization_dir/pca_evol_gif_clip_noise_config41.gif',
    #     # out_path='./visulaization_dir/pca_evol_gif_dino_noise_config1.gif',
    #     # out_path='./visulaization_dir/pca_evol_gif_dino_noise_config3.gif',
    #     # out_path='./visulaization_dir/pca_evol_gif_clip_poison_config1.gif',
    #     # out_path='./visulaization_dir/pca_evol_gif_clip_poison_config2.gif',
    #     # out_path='./visulaization_dir/pca_evol_gif_dino_poison_config1_centroids.gif',
    #     out_path='./visulaization_dir/pca_evol_gif_dino_poison_config1_points.gif',
    #     total_duration=10,
    #     dpi=300,
    #     strip_axes=False,
    # )
    
    # figures_to_frames(
    #     figs,
    #     out_dir='./visulaization_dir/pca_evol_gif_clip_noise_config28/',
    #     # out_dir='./visulaization_dir/pca_evol_gif_clip_noise_config41.gif',
    #     # out_dir='./visulaization_dir/pca_evol_gif_dino_noise_config1.gif',
    #     # out_dir='./visulaization_dir/pca_evol_gif_dino_noise_config3.gif',
    #     # out_dir='./visulaization_dir/pca_evol_gif_clip_poison_config1.gif',
    #     # out_dir='./visulaization_dir/pca_evol_gif_clip_poison_config2.gif',
    #     # out_dir='./visulaization_dir/pca_evol_gif_dino_poison_config1_centroids.gif',
    #     # out_dir='./visulaization_dir/pca_evol_gif_dino_poison_config1/',
    #     dpi=150,
    #     strip_axes=True,
    #     start_index=1
    # )
    
    
    ## CLIP NOISE
    # pickle_path_1 = 'results/single_experiment/clip_noise_TA/config28/embedding_plots/pca_alpha_60_figs.pkl'
    # alpha_1 = 2.0
    # nums_1 = np.round(np.linspace(0.0, alpha_1, 30), 2)
    # pickle_path_2 = 'results/single_experiment/clip_noise_TA/config41/embedding_plots/pca_alpha_60_figs.pkl'
    # alpha_2 = 3.5
    # nums_2 = np.round(np.linspace(0.0, alpha_2, 30), 2)
    
    ## CLIP POISON
    # pickle_path_1 = 'results/single_experiment/clip_poison_TA/config1/embedding_plots/pca_alpha_60_figs.pkl'
    # alpha_1 = 0.45
    # nums_1 = np.round(np.linspace(0.0, alpha_1, 30), 2)
    # pickle_path_2 = 'results/single_experiment/clip_poison_TA/config2/embedding_plots/pca_alpha_60_figs.pkl'
    # alpha_2 = 1.0
    # nums_2 = np.round(np.linspace(0.0, alpha_2, 30), 2)
    
    ## DINO NOISE
    # pickle_path_1 = 'results/single_experiment/dino_noise_TA/config1/embedding_plots/pca_alpha_60_figs.pkl'
    # alpha_1 = 1.35
    # nums_1 = np.round(np.linspace(0.0, alpha_1, 30), 2)
    # pickle_path_2 = 'results/single_experiment/dino_noise_TA/config3/embedding_plots/pca_alpha_60_figs.pkl'
    # alpha_2 = 2.7
    # nums_2 = np.round(np.linspace(0.0, alpha_2, 30), 2)
    
    
    ## DINO NOISE POISON
    # pickle_path_1 = 'results/single_experiment/dino_noise_TA/config3/embedding_plots/pca_alpha_60_figs.pkl'
    # alpha_1 = 2.7
    # nums_1 = np.round(np.linspace(0.0, alpha_1, 30), 2)
    # pickle_path_2 = 'results/single_experiment/dino_poison_TA/config1/embedding_plots/pca_alpha_60_figs_centroids.pkl'
    # alpha_2 = 2.05
    # nums_2 = np.round(np.linspace(0.0, alpha_2, 30), 2)

    # with open(pickle_path_1, 'rb') as f:
    #     figs1 = pickle.load(f)
    # for f in figs1:
    #     plt.close(f)
    # figs1 = figs1[0:60:2]
    # with open(pickle_path_2, 'rb') as f:
    #     figs2 = pickle.load(f)
    # for f in figs2:
    #     plt.close(f)
    # figs2 = figs2[0:60:2]
    
    
    from matplotlib import font_manager
    ttf = font_manager.findfont("DejaVu Sans")  
    
    # figures_to_frames(
    #     figs_a=figs1,
    #     figs_b=figs2,
    #     nums_a=nums_1,
    #     nums_b=nums_2,
    #     number_fmt="α={:.2f}",
    #     # out_dir='./visulaization_dir/pca_evol_gif_clip_noise_configs_28_41/',
    #     # out_dir='./visulaization_dir/pca_evol_gif_clip_poison_configs_1_2/',
    #     # out_dir='./visulaization_dir/pca_evol_gif_dino_noise_configs_1_3/',
    #     out_dir='./visulaization_dir/pca_evol_gif_dino_noise_poison_configs_3_1/',
    #     dpi=150,
    #     strip_axes=True,
    #     start_index=1,
    #     border_px=1,
    #     text_margin=(15, 8),
    #     font_size=40,
    #     font_path=ttf
    # )
    
    
    
    pickle_path_1 = 'results/single_experiment/clip_noise_TA/config28/embedding_plots/pca_alpha_60_figs.pkl'
    
    with open(pickle_path_1, 'rb') as f:
        figs1 = pickle.load(f)
    for f in figs1:
        plt.close(f)
    figs1 = figs1[0:60:20]
    
    alpha_1 = 2.0
    nums_1 = np.round(np.linspace(0.0, alpha_1, 4), 2)
    
    figures_to_frames(
        figs_a=figs1,
        nums_a=nums_1,
        number_fmt="α={:.2f}",
        out_dir='./visulaization_dir/pca_evol_gif_clip_noise_configs_28_4frames/',
        dpi=150,
        strip_axes=True,
        start_index=1,
        border_px=1,
        text_margin=(15, 8),
        font_size=40,
        font_path=ttf
    )
    