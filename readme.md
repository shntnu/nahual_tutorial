# nahual tutorial

Get image embeddings from pretrained ML models — MorphEm, DINOv2, Cellpose, etc. — through a single uniform Python API using [`nahual`](https://github.com/afermg/nahual).

`morphem_nb.py` is a marimo notebook that fetches 20 Cell Painting images, pushes them through both **MorphEm** and **DINOv2**, and compares the resulting embeddings.

| Notebook | Preview |
|---|---|
| [`morphem_nb.py`](morphem_nb.py) | [![Open in molab](https://marimo.io/molab-shield.svg)](https://molab.marimo.io/github/afermg/nahual_tutorial/blob/master/morphem_nb.py) |

The molab preview renders the committed session snapshot in [`__marimo__/session/`](__marimo__/session/) — you see the cached outputs without needing GPUs or a running nahual server.

## What is nahual?

`nahual` is an IPC-based dispatcher for ML models. Each model runs as a long-lived **server** in its own Nix/Python environment, listening on a UNIX socket (`ipc:///tmp/...`). Your **client** code talks to it over the socket with a uniform `(setup, process)` interface.

You get:
- Models hosted in isolated environments (no torch in your notebook, no dependency conflicts between models)
- Swap models without restarting your notebook (just change the socket address)
- Multiple models run side-by-side, each pinned to a different GPU

The cost: you start one server per model.

## Prerequisites

- Linux with Nix (the model servers ship as Nix flakes)
- At least one CUDA GPU
- Python 3.13 with [`uv`](https://docs.astral.sh/uv/) for the client side
- ~5 GB free disk for model weights and Nix store entries

## Quick start

### 1. Start the model servers

Open one terminal per model. Use a per-user socket name (`${USER}`) so you don't collide with other users on a shared machine:

```bash
# Terminal A — MorphEm
nix run github:afermg/nahual_vit "ipc:///tmp/morphem_${USER}.ipc"

# Terminal B — DINOv2
nix run github:afermg/dinov2 -- "ipc:///tmp/dinov2_${USER}.ipc"
```

The first run fetches the flake from GitHub and downloads model weights — expect 1–5 minutes. Subsequent runs are instant. While idle, the servers print `Waiting for Model: Timed out` — that's normal.

### 2. Launch the notebook

```bash
uv run marimo edit morphem_nb.py
```

Marimo opens in your browser. The notebook will pick up your `${USER}`-scoped socket addresses if you edited `address` / `dino_address` to match, or you can keep the defaults (`ipc:///tmp/morphem.ipc`, `ipc:///tmp/dinov2_shsingh.ipc`).

### SSH / remote

If you're SSH'd into the server, port-forward marimo from your laptop:

```bash
# on your laptop, in a new terminal
ssh -N -L 2724:localhost:2724 <server>
```

then open `http://localhost:2724` in your browser.

## The client API in one screen

Every model uses the same three calls:

```python
from nahual.process import dispatch_setup_process

# 1. Pick a model group; get its (setup, process) pair
setup, process = dispatch_setup_process("dinov2")  # or "vit", "cellpose", ...
address = "ipc:///tmp/dinov2_alice.ipc"

# 2. Load a model server-side
response = setup(dict(model_name="dinov2_vitb14"), address=address)
# response describes what the server expects (channels, tile size)

# 3. Run inference
embeddings = process(data, address=address)
# data shape: (batch, channels, z, y, x); embeddings: (batch, embedding_dim)
```

`dispatch_setup_process(name)` looks up the model group in `nahual.process.get_output_signature`. Known groups:

| name | what it is | model_name examples |
|---|---|---|
| `vit` | Vision Transformers (MorphEm, etc.) | `CaicedoLab/MorphEm` |
| `dinov2` | DINOv2 from torch.hub | `dinov2_vits14`, `dinov2_vitb14`, `dinov2_vitl14` |
| `cellpose` | Cellpose segmentation | `cyto3`, `nuclei` |
| `subcell` | SubCell segmentation | — |
| `trackastra` | Cell tracker | — |
| `recursionpharma/OpenPhenom` | OpenPhenom | — |

Any name not in the table is treated as `("dict", "numpy")` (dict-in, numpy-out).

## Per-model gotchas

All servers share `nahual.preprocess` helpers. **The notebook prints the source of these helpers live**, so the rules below are authoritative regardless of which version of `nahual` you have installed.

**Channels.** `pad_channel_dim(pixels, expected_channels)` pads the channel dim with zeros up to `expected_channels`. It does **not** truncate.

- Always send `input_channels <= expected_channels`.
- MorphEm: `expected_channels = 6` → 1, 3, 6 work. 7 will fail.
- DINOv2: `expected_channels = 3` → 1, 2, 3 work. 6 will pass through unchanged and crash the model's first conv (which expects exactly 3).

The notebook handles this by broadcasting the 1-channel Cell Painting image to 6 channels for MorphEm, and slicing the first 3 channels for DINOv2.

**Spatial dimensions.** `validate_input_shape(input_yx, expected_tile_size)` asserts that every spatial dim is divisible by the model's patch size.

- MorphEm: `tile_size = 16` → 224, 256, 512 all work.
- DINOv2: `tile_size = 14` → 224, 252, 266 work; 256 does **not**.

The notebook center-crops 256→224 before sending to DINOv2.

**Z dimension.** The wire shape is always 5D `(batch, channels, z, y, x)`. Servers drop the z axis with `pixels[:, :, 0]`, so set `z=1` for non-stacks. 4D inputs `(batch, channels, y, x)` will fail shape validation.

**Device.** `setup(dict(..., device=N), ...)` pins the model to GPU `N`. Default is `0`. If that GPU is full, inference will silently fail (see Troubleshooting).

**One model per server.** A server hosts one model at a time. Sending a new `setup` dict reloads with a new model; the previous one is gone. To run two models at the same time, start two servers on two sockets.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `struct.error: unpack requires a buffer of 2 bytes` on client | Server caught an exception and returned `{}`. Check the server terminal for `Data processing failed: <error>` or `Model loading failed: <error>`. |
| Setup returns `{}` instead of model info | `model_name` not found. DINOv2 wants torch.hub names (`dinov2_vits14`), not Hugging Face names (`facebook/dinov2-small`). |
| Inference silently OOMs | GPU 0 is full. Pass `device=N` to setup. Check `nvidia-smi`. |
| Empty / black row in cosine similarity | Image is near-uniform (empty well, edge artifact). Real biological signal, not a pipeline bug. |
| `Permission denied` on someone else's socket | IPC sockets are 755. Use your own at `ipc:///tmp/<model>_${USER}.ipc`. |

## Related

- nahual source: https://github.com/afermg/nahual
- Per-model server flakes: https://github.com/afermg/nahual_vit, https://github.com/afermg/dinov2
- Single-file example for each model: https://github.com/afermg/nahual/tree/master/examples
