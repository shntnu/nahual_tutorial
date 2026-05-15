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
    nix develop --command bash -c "python src/vit/morphem.py ipc:///tmp/morphem.ipc"
    ```

    The notebook talks to that process over the IPC socket below.
    """)
    return


@app.cell
def _(dispatch_setup_process):
    setup, process = dispatch_setup_process("vit")
    address = "ipc:///tmp/morphem.ipc"
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
    tile_size = 256
    input_shape = (2, 6, 1, tile_size, tile_size)
    data = numpy.random.random_sample(input_shape)
    data.shape
    return (data,)


@app.cell
def _(address, data, process):
    result = process(data, address=address)
    result
    return


if __name__ == "__main__":
    app.run()
