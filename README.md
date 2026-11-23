# UTRA R&D Robot Tracker

Real time detector and tracker for small student rovers in an 8 ft by 4 ft maze, seen by an overhead camera.

- Zero shot detection with an open vocabulary [YOLOE](https://docs.ultralytics.com/models/yoloe/) model and text prompts, so no labelled data is needed yet.
- `utra-track prepare` bakes the prompts into `models/robot_detector.pt` once. Runtime then needs no text encoder and no network.
- ByteTrack gives each robot a stable track ID.
- Every run writes `runs/<timestamp>/detections.csv` and `run.json`.
- Swapping in a fine tuned model is a config change: point `model.weights` at your `.pt`.
- An autograder follows the rover from the start zone to the end zone of the maze and writes PASS or FAIL. It needs no judge input. Wall contacts can be tracked and reported too.

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
| `utra-track calibrate` | Click the 4 maze corners once so pixels map to feet. Flags: `--source`, `--output` |
| `utra-track cameras` | Probe camera indices 0 to 9 and print the ones that open with their resolution |
| `utra-track check` | Print Python, torch, CUDA, GPU, whether CUDA works, Ultralytics version and whether the model file exists |

`run` also takes `--team`, `--run`, `--time-limit`, `--collisions` / `--no-collisions` and `--grading` / `--no-grading`.

Press `q` or `Esc` in the window to stop, or Ctrl+C in the terminal. Both close the camera and flush the CSV.

## Autograding a run

The overhead camera must see the whole 8 ft by 4 ft maze. Each run of the program grades one run of one robot, with no judge input.

1. **Calibrate once per camera mount.** Run `utra-track calibrate`, then click the four inside floor corners of the arena in this order:
   1. the loading zone corner
   2. the far corner along the 8 ft side
   3. the corner diagonal from the loading zone
   4. the last corner
   
   The maze walls are drawn over the image so you can check them, and Enter saves `config/arena_calibration.json`. Without calibration the whole frame is taken as the maze and the dashboard shows a red warning.
2. **Start it:** `python start.py --team 7 --run 1`.
3. Place the robot in the start zone. The clock starts when it moves 0.25 ft, and that spot is marked **A**.
4. The run ends by itself when either of these happens:
   - **settled in end zone:** the robot has stayed still for `settle_s` (2 s) with at least half its outline inside the end zone. Driving through an end cell does not end the run.
   - **time limit:** `time_limit_s` (8 min) runs out.

   Pressing `q`, Ctrl+C or the video ending also stops it, and the last seen position counts as the end.
5. The run is graded:
   - **PASS** means at least half the outline was in the start zone when it set off, and at least half was in the end zone at the end.
   - Anything else is **FAIL**.

   PASS or FAIL shows on screen for 3 seconds, then the program exits with code 0 for PASS and 1 for FAIL, so it can be scripted.

The zones default to the loading zone `L` for the start and any `B` cell for the end. To change them, set `grading.start_zone` and `grading.end_zone` to layout letters, or to a list of `[col, row]` cells:

```yaml
grading:
  start_zone: L
  end_zone: [[7, 3]]   # only the bottom right B
```

Columns count 0 to 7 from the loading zone side along the 8 ft wall. Rows count 0 to 3 from the loading zone side along the 4 ft wall. You can also mark cells `S` or `E` in `arena.layout` and use `start_zone: S`, `end_zone: E`.

The keys are for the operator only: `c` turns wall contact tracking on or off, `r` resets the run, `h` hides the help and `q` quits.

**Wall collisions** are optional and only reported. They never change PASS or FAIL. The rover outline comes from the segmentation mask, or from the box with `grading.footprint: box`. It is mapped to maze feet and checked against the block faces and outer walls:

- A contact counts after `contact_frames` frames deeper than `contact_margin_ft`.
- It ends after `release_frames` clear frames.
- It is **major** if it goes `major_depth_ft` deep or lasts `major_duration_s`. Otherwise it is **minor**.

To turn tracking off, set `grading.collisions: false`, pass `--no-collisions`, or press `c`. The camera sees the rover's top plate, which looks shifted outward near the edges of the image. If you get false contacts near the outer walls, set `arena.camera_height_ft` or raise `contact_margin_ft`.

Each run writes these files to its folder:

- `result.json`: outcome, why the run ended, time, start and end cell, how much of the rover was inside each zone, distance, 1 ft steps and contacts
- `track.csv`: position in feet for every frame
- `path.png`: a top down map of the path with the zones, start, end and contacts marked. It shows no people

Every run also adds one row to `runs/results.csv`, so all teams end up in one table.

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
| `arena.layout` | Rubric maze | One string per row of 1 ft cells: `.` open, `#` block, `L` loading zone, `B` drop off, `S` / `E` spare zone letters |
| `arena.calibration` | `config/arena_calibration.json` | Written by `calibrate` |
| `arena.camera_height_ft` / `robot_height_ft` | `null` / `0.5` | Parallax correction, off while the camera height is null |
| `grading.enabled` | `true` | Autograder on, `false` shows the plain tracker |
| `grading.start_zone` / `end_zone` | `L` / `B` | Layout letters or a list of `[col, row]` cells |
| `grading.zone_share` | `0.5` | Share of the outline that must be inside a zone |
| `grading.start_move_ft` | `0.25` | Movement that starts the clock |
| `grading.settle_s` | `2.0` | Seconds still in the end zone that end the run |
| `grading.time_limit_s` | `480` | Run ends as FAIL after this, `null` for no limit |
| `grading.collisions` | `true` | Wall contact tracking, reported only |
| `grading.footprint` | `mask` | `mask` or `box` outline for contact checks |
| `grading.contact_margin_ft` | `0.04` | Contacts shallower than this are ignored |
| `grading.major_depth_ft` / `major_duration_s` | `0.15` / `1.0` | When a contact counts as major |
| `grading.contact_frames` / `release_frames` | `3` / `6` | Debounce |

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

1. Fine tune on our own overhead footage and ship the result as `model.weights`. Good masks make contact detection much more reliable.
2. Find the maze corners automatically from AprilTags instead of clicking them.
3. Check the contact thresholds against hand counts from real runs.
