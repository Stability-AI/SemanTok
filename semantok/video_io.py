"""Write clips ``[3, T, H, W]`` in [-1, 1] as an mp4 grid."""
import torch


def write_grid(path, rows, fps=8, pad=4):
    """``rows``: list of rows, each a list of clips with the same shape."""
    import imageio

    C, T, H, W = rows[0][0].shape
    nr, nc = len(rows), max(len(r) for r in rows)
    grid = torch.ones(T, nr * H + (nr - 1) * pad, nc * W + (nc - 1) * pad, 3)
    for i, row in enumerate(rows):
        for j, clip in enumerate(row):
            y, x = i * (H + pad), j * (W + pad)
            grid[:, y:y + H, x:x + W] = ((clip.float().cpu() + 1) / 2).clamp(0, 1).permute(1, 2, 3, 0)
    frames = (grid * 255).round().byte().numpy()
    imageio.mimwrite(path, list(frames), fps=fps, macro_block_size=1)
