try:
    from .plot_2d import nx_draw_with_pos_and_attr
except ImportError:
    # matplotlib not available
    def nx_draw_with_pos_and_attr(*args, **kwargs):
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")
