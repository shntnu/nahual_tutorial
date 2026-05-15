import marimo

__generated_with = "0.23.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy

    from nahual.process import dispatch_setup_process

    return dispatch_setup_process, mo, numpy


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # MorphEm via nahual

    This notebook drives the [MorphEm](https://huggingface.co/CaicedoLab/MorphEm)
    ViT model through a `nahual` server running in a separate environment.

    **Prerequisites:** clone [`nahual_vit`](https://github.com/afermg/nahual_vit)
    and start the server in its own shell:

    ```bash
    nix run github:afermg/nahual_vit "ipc:///tmp/morphem_{NAME}.ipc"
    ```


    The notebook talks to that process over the IPC socket below.
    """)
    return


@app.cell
def _(dispatch_setup_process):
    setup, process = dispatch_setup_process("vit")
    address = "ipc:///tmp/morphem_alan.ipc"
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Build a random input batch

    MorphEm expects shape `(batch, channels, depth, H, W)`. The server pads
    when `channels < 6`, so any value up to 6 works.
    """)
    return


@app.cell
def _(numpy):
    import io
    import imagecodecs
    import requests

    tile_size = 256

    # data = numpy.random.random_sample(input_shape)

    url = (
        "https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_4/"
        "images/2021_04_26_Batch1/images/"
        "BR00117035__2021-05-02T16_02_51-Measurement1/Images/"
        "r01c01f01p01-ch1sk1fk1fl1.tiff"
    )
    raw = imagecodecs.imread(io.BytesIO(requests.get(url, timeout=30).content))
    img = raw.astype("float32") / numpy.iinfo(raw.dtype).max

    # Tile the (H, W) image into non-overlapping tile_size x tile_size patches.
    h_tiles = img.shape[0] // tile_size
    w_tiles = img.shape[1] // tile_size
    cropped = img[: h_tiles * tile_size, : w_tiles * tile_size]
    tiles = cropped.reshape(h_tiles, tile_size, w_tiles, tile_size).swapaxes(1, 2)
    tiles = tiles.reshape(h_tiles * w_tiles, tile_size, tile_size)

    # Expand to (batch, channels=6, depth=1, H, W); repeat the single channel.
    batch = tiles.shape[0]
    input_shape = (batch, 6, 1, tile_size, tile_size)
    data = numpy.broadcast_to(tiles[:, None, None, :, :], input_shape).copy()
    data.shape

    return (data,)


@app.cell
def _(address, data, process):
    result = process(data, address=address)
    result
    return


if __name__ == "__main__":
    app.run()
