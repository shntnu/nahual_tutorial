# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A short tutorial showing how to call pretrained vision models (MorphEm, DINOv2) through [`nahual`](https://github.com/afermg/nahual). The deliverable is one marimo notebook (`morphem_nb.py`) that fetches 20 JUMP Cell Painting images, embeds them with both models, and compares the embedding geometries side-by-side. `morphem.py` is a much shorter single-file example kept for reference.

There is no test suite, no CI, no build step. Edits to the notebook are the work.

## Platform: Linux+Nix is canonical; pixi is a portable alternative

The Linux+Nix+CUDA path described in the README is the canonical one and stays untouched. A parallel pixi-based path was added that also works on macOS (MPS) and on NixOS via GPU. Verified end-to-end:

- aarch64-darwin (MPS): `morphem (20, 2304)`, `dinov2 (20, 768)`. Norm-means 145.79 / 47.76.
- oppy (NixOS+H100, pixi path): `morphem (20, 2304)` in 0.45s, `dinov2 (20, 768)` in 0.31s. Norm-means 145.57 / 47.78. `device='cuda:0'`.

**Why pixi rather than fixing the Nix flake on Darwin:** nixpkgs marks `python3.13-torch` as broken on Darwin, so `nix run github:afermg/nahual_vit` and `nix run github:afermg/dinov2` both fail at Nix *evaluation*. The torch derivation itself is the blocker, not the flake. Conda-forge ships PyTorch with MPS on Darwin and CUDA on Linux from one channel, so pixi is the pragmatic fix.

**Pixi has two environments per server** because CUDA system-requirements cannot coexist with macOS in a single env:

- `default` (Linux+CUDA via conda-forge `pytorch-gpu` + `system-requirements.cuda = "12"`). Used with `pixi run morphem ...`.
- `osx` (macOS+MPS via conda-forge `pytorch`). Used with `pixi run -e osx morphem ...`.

**On NixOS specifically**, conda-installed `pytorch-gpu` can't find `libcuda.so` because `/run/opengl-driver/lib` isn't on the default linker path. Each server flake now exposes a minimal `devShells.pixi` that bootstraps `pixi` with the right `LD_LIBRARY_PATH`. Pattern lifted from [shntnu/neusis templates/python-pixi](https://github.com/shntnu/neusis/tree/main/templates/python-pixi). Use as:

```bash
cd ../nahual_vit && nix develop .#pixi --command pixi run morphem "ipc:///tmp/morphem_${USER}.ipc"
cd ../dinov2     && nix develop .#pixi --command pixi run dinov2  "ipc:///tmp/dinov2_${USER}.ipc"
```

**Where the pixi files live.** Not in this repo - inside the sibling clones of the server repos:

- `../nahual_vit/pixi.toml` + new `devShells.pixi` in `flake.nix`
- `../dinov2/pixi.toml`     + new `devShells.pixi` in `flake.nix`

**Server-code changes that went with the pixi path** (also in the sibling clones, not here):

- `nahual_vit/src/vit/setup.py`: added `_resolve_device(device)` - strings pass through, ints select CUDA→MPS→CPU. Server tolerates `device=2` from a Mac client.
- `dinov2/server.py`: dropped `assert torch.cuda.is_available()`, added the same `_resolve_device`, removed a stray `.cuda()` call before `.to(device)`.

**Notebook change in this repo.** `morphem_nb.py` detects Darwin and sends `device="mps"` for both servers; Linux+CUDA still gets the original `device=2` / `device=3` indices.

**Version pin to remember.** MorphEm's HuggingFace remote-code `vision_transformer.py` reads `VisionTransformer.all_tied_weights_keys`, which was removed in `transformers` 5.x. The pixi env pins `transformers >=4.57.3,<5`. The Nix flake gets away with a permissive bound because nixpkgs ships an older transformers; pixi/conda-forge will happily resolve 5.x without a cap.

**Nahual must come from git, not PyPI** in both pixi envs. The server-side `nahual.preprocess.channel_chunks_rigid3` (used by dinov2) only exists on master; PyPI 0.0.8 lags. Both `pixi.toml`s declare `nahual = { git = "https://github.com/afermg/nahual.git" }`.

## Commands

```bash
# enter the Nix dev shell (provides Python 3 + uv + the C libs torch needs at runtime)
nix develop                       # or: direnv allow (an .envrc is checked in)

# install/refresh client deps from uv.lock
uv sync

# edit the notebook
uv run marimo edit morphem_nb.py

# run the notebook headless (regenerates __marimo__/session/ snapshot)
uv run marimo run morphem_nb.py
```

The notebook is a **client**. It will hang on `setup(...)` unless two model servers are already running on their IPC sockets. Two ways to start them, each in its own terminal:

```bash
# Linux+CUDA (canonical):
nix run github:afermg/nahual_vit -- "ipc:///tmp/morphem_${USER}.ipc"
nix run github:afermg/dinov2     -- "ipc:///tmp/dinov2_${USER}.ipc"

# macOS (pixi sidecar, run from sibling clones; osx env):
cd ../nahual_vit && pixi run -e osx morphem "ipc:///tmp/morphem_${USER}.ipc"
cd ../dinov2     && pixi run -e osx dinov2  "ipc:///tmp/dinov2_${USER}.ipc"

# NixOS pixi+GPU (default env; nix shell supplies LD_LIBRARY_PATH for libcuda):
cd ../nahual_vit && nix develop .#pixi --command pixi run morphem "ipc:///tmp/morphem_${USER}.ipc"
cd ../dinov2     && nix develop .#pixi --command pixi run dinov2  "ipc:///tmp/dinov2_${USER}.ipc"
```

The notebook builds socket addresses from `getpass.getuser()`, so it picks up the `${USER}`-scoped sockets automatically regardless of which launcher you used.

## Architecture: client / server split via IPC

`nahual` is a dispatcher, not a model library. Three things to internalise before editing:

1. **The notebook never imports `torch`.** Each model lives in its own Nix-managed Python environment as a long-lived server process listening on a UNIX socket. The client (this notebook) talks to it over the socket with a uniform `(setup, process)` pair returned by `nahual.process.dispatch_setup_process(name)`.

2. **One model per socket.** A server hosts exactly one model at a time. Sending a new `setup` dict reloads the model and evicts the previous one. To run MorphEm and DINOv2 simultaneously (as the notebook does), start two servers on two sockets and pin them to different GPUs via `setup(dict(..., device=N), ...)`. The notebook uses `device=2` for MorphEm and `device=3` for DINOv2 — adjust to whatever is free on the target machine.

3. **The wire format is fixed: 5D `(batch, channels, z, y, x)` numpy in, 2D `(batch, embedding_dim)` numpy out.** 4D inputs fail shape validation server-side. Set `z=1` for non-stacks; servers drop the z axis with `pixels[:, :, 0]`.

The server's `setup` response is a **self-describing contract**: it returns `expected_channels` and `expected_yx` for the loaded model. Read these instead of hardcoding shapes. The notebook deliberately prints the source of `nahual.preprocess.pad_channel_dim` and `validate_input_shape` so the rules below are authoritative against whatever `nahual` version is installed.

## Shape invariants (these will bite you)

- **Channels.** `pad_channel_dim` pads with zeros up to `expected_channels` but does **not** truncate. Send `input_channels <= expected_channels`. MorphEm wants 6; DINOv2 wants 3. If you send 6 to DINOv2 it passes through unpadded and crashes the model's first conv. The notebook broadcasts the single Cell Painting channel to 6 for MorphEm, then slices `data[:, :3, ...]` for DINOv2.
- **Spatial.** `validate_input_shape` asserts every spatial dim is divisible by the model's patch size. MorphEm patch=16 (256 OK); DINOv2 patch=14 (256 NOT OK, 224 OK). The notebook center-crops 256→224 (`16:240`) before sending to DINOv2.
- **Silent OOM.** If the pinned GPU is full, the server catches the exception and returns `{}`; the client then dies with a cryptic `struct.error: unpack requires a buffer of 2 bytes`. Check `nvidia-smi` and the server's terminal output before debugging the client.

## The molab snapshot

`__marimo__/session/` holds cached cell outputs that let molab render the notebook without a running nahual server or GPU. It is checked in. Re-run `marimo run` (or click "Save snapshot" in the marimo UI) after meaningful changes so the molab preview stays current; otherwise the README badge will show stale outputs.

## Conventions

- Python 3.12+ via `uv` (locked in `uv.lock`). Don't switch to pip / conda / poetry.
- The Nix flake only provides the C runtime libs (`zlib`, `libstdc++`) that wheels installed by `uv` link against — it does not manage Python deps. `uv` owns the Python environment; `nix develop` just makes torch's wheels loadable.
- Cell Painting is a proper noun, title-case it. Domain terms (mAP, well, plate, perturbation, JUMP) are assumed vocabulary.
