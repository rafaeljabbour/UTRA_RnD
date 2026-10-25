# UTRA R&D Robot Tracker

Real time detector and tracker for small student rovers in an 8 ft by 4 ft maze, seen by an overhead camera.

- Zero shot detection with an open vocabulary [YOLOE](https://docs.ultralytics.com/models/yoloe/) model and text prompts, so no labelled data is needed yet.
- `utra-track prepare` bakes the prompts into `models/robot_detector.pt` once. Runtime then needs no text encoder and no network.
- ByteTrack gives each robot a stable track ID.
- Every run writes `runs/<timestamp>/detections.csv` and `run.json`.
- Swapping in a fine tuned model is a config change: point `model.weights` at your `.pt`.

## Quick start

Needs Python 3.10 or newer and internet for the first install. git is recommended but optional.

```
python install.py
python start.py
```

`install.py` creates `.venv`, installs PyTorch, installs this package, builds the model and runs `utra-track check`. It is safe to rerun.

The PyTorch wheel is picked from the GPU that `nvidia-smi` reports. Turing and newer cards get `cu128`. Maxwell, Pascal and Volta cards such as the Quadro M4000 get `cu126` with `torch<2.15`, since newer wheels no longer run on them. Kepler and older cards, and PCs without an NVIDIA GPU, get CPU PyTorch. After installing, a tiny CUDA test runs and, if it fails, the installer retries once with `cu126` before warning that tracking will run on CPU.

Options: `--cpu`, `--cuda <tag>` or the `UTRA_TORCH_CUDA` env var to override the automatic choice, `--reinstall-torch`, `--skip-prepare`.

If `utra-track check` shows `cuda works   no`, run `python3 install.py --reinstall-torch`. Until then `detect.device: auto` falls back to CPU with a warning.

`start.py` forwards any arguments to `utra-track run`, for example `python start.py --source 1 --no-show`.

## Platform notes

- **Windows**: use `python` or `py` from cmd or PowerShell. No activation scripts are involved, so the execution policy does not matter. Cameras open with DirectShow for fast startup.
- **Linux**: use `python3`. Your user may need to be in the `video` group to read `/dev/video*`.
- **Jetson**: needs the NVIDIA JetPack PyTorch wheels, which `install.py` does not handle yet. Install torch from NVIDIA first, then `pip install -e ".[prepare]"` and `utra-track prepare`.

## Commands

| Command | What it does |
| --- | --- |
| `utra-track prepare [--force]` | Download base weights, bake the prompt classes, save `model.weights` |
| `utra-track run` | Detect and track. Flags: `--source`, `--config`, `--conf`, `--no-show`, `--save-video`, `--max-frames` |
| `utra-track cameras` | Probe camera indices 0 to 9 and print the ones that open with their resolution |
| `utra-track check` | Print Python, torch, CUDA, GPU, whether CUDA works, Ultralytics version and whether the model file exists |

Press `q` or `Esc` in the window to stop, or Ctrl+C in the terminal. Both close the camera and flush the CSV.

## Config

`config/default.yaml` holds everything. CLI flags override it.

| Key | Default | Notes |
| --- | --- | --- |
| `source` | `0` | Camera index, video file, or `rtsp://` or `http://` URL |
| `model.weights` | `models/robot_detector.pt` | Model used at runtime |
| `model.base` | `yoloe-26m-seg.pt` | Open vocabulary base, only used by `prepare` |
| `model.classes` | `white disk robot`, `mobile robot`, `robot car` | Prompts. Rerun `prepare --force` after changing |
| `detect.conf` / `iou` / `imgsz` | `0.25` / `0.5` / `640` | |
| `detect.device` | `auto` | `cuda:0` if available and working, else `cpu` |
| `tracker` | `bytetrack.yaml` | Any Ultralytics tracker config |
| `display` | `true` | Live window with boxes, IDs and FPS |
| `output.dir` | `runs` | |
| `output.save_video` | `false` | See privacy note |

The default prompts were picked from overhead photos of our current rover, a white round top plate. Zero shot scores are modest, about 0.35 to 0.7 on the robot, so tune `detect.conf` for your lighting and expect fine tuning to help a lot.

`detections.csv` columns: `frame, time_s, track_id, class_name, conf, x1, y1, x2, y2, cx, cy` in pixels. `track_id` is `-1` for a detection the tracker has not confirmed yet.

## Privacy

Students walk around the maze and may appear on camera. Video is never written unless you pass `--save-video`, and only boxes and coordinates go to the CSV. Do not enable video recording during sessions where people are in frame.

## Development

```
pip install -e ".[dev]"
ruff check . && ruff format --check . && pytest
```

Tests need no weights, GPU or network. CI runs them on every push.

## Roadmap

1. Fine tune on our own overhead footage and ship the result as `model.weights`.
2. Map pixels to maze coordinates with a homography from AprilTags at the maze corners.
3. Collision and wall contact events from tracks and the maze layout.
