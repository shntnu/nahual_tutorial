import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import getpass

    import marimo as mo
    import numpy

    from nahual.process import dispatch_setup_process

    return dispatch_setup_process, getpass, mo, numpy


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Image embeddings via nahual

    This notebook drives two pretrained vision models — [MorphEm](https://huggingface.co/CaicedoLab/MorphEm)
    and [DINOv2](https://github.com/facebookresearch/dinov2) — through `nahual`
    model servers and compares their image embeddings on 20 Cell Painting wells.

    **See `README.md` in this repo** for prerequisites (Nix, GPU) and the exact
    commands to start the two model servers in separate shells. Once both
    servers are listening on their IPC sockets, walk through this notebook
    top-to-bottom.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## How nahual works

    Each ML model runs as a long-lived **server** in its own Nix/Python
    environment, listening on a UNIX socket (`ipc:///tmp/...`). Your **client**
    code (this notebook) asks `nahual.process.dispatch_setup_process(name)` for
    a uniform `(setup, process)` pair:

    - `setup(dict(...), address=...)` loads a model on the server.
    - `process(data, address=...)` runs inference on a numpy array.

    The wire format is the same for every model — numpy in, numpy out — so
    swapping models requires only changing the socket address. Your notebook
    never imports `torch`. The model lives in the server's environment, not
    yours.

    The cell below shows every model group that the installed `nahual` knows
    about.
    """)
    return


@app.cell(hide_code=True)
def _():
    import inspect as _inspect
    import re as _re
    from nahual.process import get_output_signature

    _src = _inspect.getsource(get_output_signature)
    _dict_block = _re.search(r"OUTPUT_SIGNATURES\s*=\s*\{([^}]*)\}", _src).group(0)
    print("Known model groups (from nahual.process.get_output_signature):")
    print(_dict_block)
    print()
    print('Anything else falls back to ("dict", "numpy") -- dict in, numpy out.')
    return


@app.cell
def _(dispatch_setup_process, getpass):
    setup, process = dispatch_setup_process("vit")
    address = f"ipc:///tmp/morphem_{getpass.getuser()}.ipc"
    return address, process, setup


@app.cell
def _(address, setup):
    # Load the MorphEm model in the server-side process.
    # `device=0` is optional — uncomment to pin to a specific GPU.
    parameters = dict(
        model_name="CaicedoLab/MorphEm",
        # device=0,
    )
    response = setup(parameters, address=address)
    response
    return (response,)


@app.cell(hide_code=True)
def _(mo, response):
    mo.md(f"""
    ### Reading the server's setup response

    The dict above is the server's **contract**: it tells you exactly what
    shapes the model accepts. For MorphEm:

    - `expected_channels = {response["execution"].get("expected_channels", "?")}`
      — see the per-model gotchas section for what happens with mismatched
      channels.
    - `expected_yx = {response["execution"].get("expected_yx", "?")}`
      — every spatial dim of the input must be divisible by this.

    This same response shape works for every nahual model (cellpose, dinov2,
    subcell, ...), so the contract is always self-describing.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build a batch of 20 separate source images

    MorphEm expects shape `(batch, channels, depth, H, W)`. We pull 20 distinct
    fields from a JUMP Cell Painting plate (`BR00117035`, wells `r02c02..r03c11`),
    center-crop each to `tile_size x tile_size`, and broadcast the single
    channel to 6. The server pads when `channels < 6`, so any value up to 6
    works.
    """)
    return


@app.cell
def _(numpy):
    import io
    import imagecodecs
    import requests
    from concurrent.futures import ThreadPoolExecutor

    tile_size = 256

    _URL = (
        "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_4/"
        "images/2021_04_26_Batch1/images/"
        "BR00117035__2021-05-02T16_02_51-Measurement1/Images/"
        "r{r:02d}c{c:02d}f01p01-ch1sk1fk1fl1.tiff"
    )
    well_positions = [(r, c) for r in (2, 3) for c in range(2, 12)]
    image_labels = [f"r{r:02d}c{c:02d}" for r, c in well_positions]

    def _fetch(rc):
        r, c = rc
        raw = imagecodecs.imread(
            io.BytesIO(requests.get(_URL.format(r=r, c=c), timeout=60).content)
        )
        arr = raw.astype("float32") / numpy.iinfo(raw.dtype).max
        _h, _w = arr.shape
        _y0, _x0 = (_h - tile_size) // 2, (_w - tile_size) // 2
        return arr[_y0:_y0 + tile_size, _x0:_x0 + tile_size]

    with ThreadPoolExecutor(max_workers=8) as _ex:
        crops = list(_ex.map(_fetch, well_positions))

    tiles = numpy.stack(crops)
    batch = tiles.shape[0]
    input_shape = (batch, 6, 1, tile_size, tile_size)
    data = numpy.broadcast_to(tiles[:, None, None, :, :], input_shape).copy()
    data.shape
    return batch, data, image_labels, tile_size, tiles


@app.cell
def _(address, data, process):
    result = process(data, address=address)
    result
    return (result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Visualize images and embeddings

    Show the 20 source-image crops, the cosine-similarity matrix between MorphEm
    embeddings, and a 2D PCA scatter.
    """)
    return


@app.cell(hide_code=True)
def _(image_labels, numpy, tile_size, tiles):
    import matplotlib.pyplot as plt

    _lo, _hi = numpy.percentile(tiles, (1, 99))
    _nrows, _ncols = 4, 5
    _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(10, 8))
    for _k, _ax in enumerate(_axes.flat):
        if _k < len(tiles):
            _ax.imshow(tiles[_k], cmap="gray", vmin=_lo, vmax=_hi)
            _ax.set_title(f"{_k}: {image_labels[_k]}", fontsize=8)
        _ax.set_axis_off()
    _fig.suptitle(f"{len(tiles)} source images (center {tile_size}x{tile_size} crops)")
    _fig.tight_layout()
    _fig
    return (plt,)


@app.cell(hide_code=True)
def _(numpy, plt, result):
    _norms = numpy.linalg.norm(result, axis=1, keepdims=True)
    _emb_n = result / numpy.clip(_norms, 1e-12, None)
    sim = _emb_n @ _emb_n.T

    _fig, _ax = plt.subplots(figsize=(6, 5))
    _im = _ax.imshow(sim, cmap="viridis", vmin=sim.min(), vmax=1.0)
    _ax.set_title("Cosine similarity between image embeddings")
    _ax.set_xlabel("image index")
    _ax.set_ylabel("image index")
    _ax.set_xticks(range(len(sim)))
    _ax.set_yticks(range(len(sim)))
    _fig.colorbar(_im, ax=_ax, fraction=0.046)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(numpy, plt, result):
    _centered = result - result.mean(axis=0, keepdims=True)
    _U, _S, _Vt = numpy.linalg.svd(_centered, full_matrices=False)
    pca_coords = _U[:, :2] * _S[:2]
    _explained = (_S ** 2) / (_S ** 2).sum()

    _fig, _ax = plt.subplots(figsize=(6, 5))
    _ax.scatter(pca_coords[:, 0], pca_coords[:, 1],
                c=numpy.arange(len(pca_coords)), cmap="tab20", s=80)
    for _k, (_x, _y) in enumerate(pca_coords):
        _ax.annotate(str(_k), (_x, _y), xytext=(5, 5),
                     textcoords="offset points", fontsize=9)
    _ax.set_xlabel(f"PC1 ({_explained[0]*100:.1f}%)")
    _ax.set_ylabel(f"PC2 ({_explained[1]*100:.1f}%)")
    _ax.set_title("MorphEm embeddings - 2D PCA")
    _fig
    return (pca_coords,)


@app.cell(hide_code=True)
def _(batch, mo, result):
    mo.md(f"""
    ## :)

    Alan revised his claim — said it would break with 20 *images* (not tiles).
    Pipeline just consumed **{batch} separate images** through MorphEm and
    returned **{result.shape}**.
    """)
    return


@app.cell(hide_code=True)
def _(address, data, numpy, plt, process):
    import time
    import datetime

    _t0 = time.time()
    proof_result = process(data, address=address)
    _dt = time.time() - _t0
    _ts = datetime.datetime.now().isoformat(timespec="seconds")

    assert proof_result.shape == (data.shape[0], 2304), (
        f"unexpected output shape {proof_result.shape}"
    )

    _n_rows, _n_cols = proof_result.shape
    _req_bytes = data.nbytes
    _resp_bytes = proof_result.nbytes
    _norms = numpy.linalg.norm(proof_result, axis=1)

    _fig, (_ax_top, _ax_bot) = plt.subplots(
        2, 1, figsize=(11, 7),
        gridspec_kw={"height_ratios": [1, 3]},
    )
    _ax_top.bar(range(_n_rows), _norms, color="steelblue")
    _ax_top.set_xticks(range(_n_rows))
    _ax_top.set_xlim(-0.5, _n_rows - 0.5)
    _ax_top.set_ylabel("L2 norm")
    _ax_top.set_title(
        f"PROOF  server@{address}  ->  shape={proof_result.shape}  "
        f"REQ={_req_bytes/1e6:.1f}MB  RESP={_resp_bytes/1e3:.1f}KB  "
        f"latency={_dt*1000:.0f}ms  @ {_ts}"
    )

    _im = _ax_bot.imshow(proof_result, aspect="auto", cmap="magma",
                          interpolation="nearest")
    _ax_bot.set_xlabel(f"embedding dimension ({_n_cols} cols)")
    _ax_bot.set_ylabel(f"tile index ({_n_rows} rows)")
    _ax_bot.set_yticks(range(_n_rows))
    _ax_bot.set_xticks([0, _n_cols // 4, _n_cols // 2, 3 * _n_cols // 4, _n_cols - 1])
    _ax_bot.hlines(numpy.arange(_n_rows) + 0.5, -0.5, _n_cols - 0.5,
                   colors="white", linewidths=0.3, alpha=0.4)
    _fig.colorbar(_im, ax=_ax_bot, fraction=0.02, pad=0.01)
    _fig.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Compare with DINOv2

    Push the same 20 images through a `dinov2_vitb14` (torch.hub) server running
    on a second IPC socket. DINOv2 has patch size 14 and expects 3 channels, so
    we take the first 3 channels and center-crop `256 -> 224` (= 16 patches per
    side) before sending.
    """)
    return


@app.cell(hide_code=True)
def _():
    import inspect as _insp
    from nahual.preprocess import pad_channel_dim, validate_input_shape

    print("### Channel-handling rule (pad_channel_dim) ###")
    print(_insp.getsource(pad_channel_dim))
    print()
    print("### Spatial-validation rule (validate_input_shape) ###")
    print(_insp.getsource(validate_input_shape))
    print()
    print("Practical consequence:")
    print("  MorphEm expects 6 channels, tile_size=16 -> send <=6 channels, H,W % 16 == 0")
    print("  DINOv2  expects 3 channels, tile_size=14 -> send <=3 channels, H,W % 14 == 0")
    print()
    print("That is why the notebook broadcasts to 6 channels and uses 256x256 for MorphEm,")
    print("then slices to 3 channels and crops to 224x224 (= 16*14) for DINOv2.")
    return


@app.cell(hide_code=True)
def _(dispatch_setup_process, getpass):
    setup_dino, process_dino = dispatch_setup_process("dinov2")
    dino_address = f"ipc:///tmp/dinov2_{getpass.getuser()}.ipc"
    return dino_address, process_dino, setup_dino


@app.cell(hide_code=True)
def _(dino_address, setup_dino):
    parameters_dino = dict(
        model_name="dinov2_vitb14",
        device=3,  # MorphEm holds GPU 0; put DINOv2 on a free GPU
    )
    response_dino = setup_dino(parameters_dino, address=dino_address)
    response_dino
    return


@app.cell(hide_code=True)
def _(data, dino_address, process_dino):
    # 3 channels, 224x224 (multiple of patch=14)
    data_dino = data[:, :3, :, 16:240, 16:240].copy()
    result_dino = process_dino(data_dino, address=dino_address)
    result_dino.shape
    return (result_dino,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Side-by-side comparison

    `MorphEm` is trained on Cell Painting; `DINOv2` is trained on natural
    images. Different feature spaces produce different similarity structure
    on the same 20 wells.
    """)
    return


@app.cell(hide_code=True)
def _(numpy, plt, result, result_dino):
    _norms_m = numpy.linalg.norm(result, axis=1, keepdims=True)
    _emb_m = result / numpy.clip(_norms_m, 1e-12, None)
    sim_morphem = _emb_m @ _emb_m.T

    _norms_d = numpy.linalg.norm(result_dino, axis=1, keepdims=True)
    _emb_d = result_dino / numpy.clip(_norms_d, 1e-12, None)
    sim_dino = _emb_d @ _emb_d.T

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))
    _im0 = _axes[0].imshow(sim_morphem, cmap="viridis",
                           vmin=min(sim_morphem.min(), sim_dino.min()), vmax=1.0)
    _axes[0].set_title(f"MorphEm cosine sim  ({result.shape[1]}-d)")
    _axes[0].set_xlabel("image index")
    _axes[0].set_ylabel("image index")
    _axes[0].set_xticks(range(len(sim_morphem)))
    _axes[0].set_yticks(range(len(sim_morphem)))
    _fig.colorbar(_im0, ax=_axes[0], fraction=0.046)

    _im1 = _axes[1].imshow(sim_dino, cmap="viridis",
                           vmin=min(sim_morphem.min(), sim_dino.min()), vmax=1.0)
    _axes[1].set_title(f"DINOv2 cosine sim  ({result_dino.shape[1]}-d)")
    _axes[1].set_xlabel("image index")
    _axes[1].set_xticks(range(len(sim_dino)))
    _axes[1].set_yticks(range(len(sim_dino)))
    _fig.colorbar(_im1, ax=_axes[1], fraction=0.046)
    _fig.tight_layout()
    _fig
    return sim_dino, sim_morphem


@app.cell(hide_code=True)
def _(numpy, plt, sim_dino, sim_morphem):
    _i, _j = numpy.triu_indices(len(sim_morphem), k=1)
    _x = sim_morphem[_i, _j]
    _y = sim_dino[_i, _j]
    _corr = numpy.corrcoef(_x, _y)[0, 1]

    _fig, _ax = plt.subplots(figsize=(6, 6))
    _ax.scatter(_x, _y, s=30, alpha=0.6)
    _ax.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.4)
    _ax.set_xlabel("MorphEm pair similarity")
    _ax.set_ylabel("DINOv2 pair similarity")
    _ax.set_title(f"Per-image-pair similarity agreement  (Pearson r = {_corr:.2f})")
    _ax.set_aspect("equal")
    _fig
    return


@app.cell(hide_code=True)
def _(image_labels, numpy, pca_coords, plt, result_dino):
    _centered_d = result_dino - result_dino.mean(axis=0, keepdims=True)
    _Ud, _Sd, _Vtd = numpy.linalg.svd(_centered_d, full_matrices=False)
    pca_coords_dino = _Ud[:, :2] * _Sd[:2]
    _exp_d = (_Sd ** 2) / (_Sd ** 2).sum()

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 5))
    _axes[0].scatter(pca_coords[:, 0], pca_coords[:, 1],
                     c=numpy.arange(len(pca_coords)), cmap="tab20", s=80)
    for _k, (_x, _y) in enumerate(pca_coords):
        _axes[0].annotate(image_labels[_k], (_x, _y), xytext=(5, 5),
                          textcoords="offset points", fontsize=7)
    _axes[0].set_title("MorphEm - 2D PCA")
    _axes[0].set_xlabel("PC1")
    _axes[0].set_ylabel("PC2")

    _axes[1].scatter(pca_coords_dino[:, 0], pca_coords_dino[:, 1],
                     c=numpy.arange(len(pca_coords_dino)), cmap="tab20", s=80)
    for _k, (_x, _y) in enumerate(pca_coords_dino):
        _axes[1].annotate(image_labels[_k], (_x, _y), xytext=(5, 5),
                          textcoords="offset points", fontsize=7)
    _axes[1].set_title("DINOv2 - 2D PCA")
    _axes[1].set_xlabel("PC1")
    _axes[1].set_ylabel("PC2")
    _fig.suptitle("Same 20 images, two models, two embedding geometries")
    _fig.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
