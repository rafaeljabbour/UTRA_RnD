# UTRA R&D Robot Tracker

Real time detector and tracker for small student rovers in an 8 ft by 4 ft maze, seen by an overhead camera.

- Zero shot detection with an open vocabulary [YOLOE](https://docs.ultralytics.com/models/yoloe/) model and text prompts, so no labelled data is needed yet.
- `utra-track prepare` bakes the prompts into `models/robot_detector.pt` once. Runtime then needs no text encoder and no network.
- ByteTrack gives each robot a stable track ID.
- Every run writes `runs/<timestamp>/detections.csv` and `run.json`.
- Swapping in a fine tuned model is a config change: point `model.weights` at your `.pt`.
- A grading dashboard follows the rover from its start to its end point in the MIE444 maze. It logs wall contacts if you want it to, and fills in the Milestone 2 or 3 rubric items the camera can judge. The judge marks the rest.

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
| `utra-track grade <run dir>` | Walk through every rubric item, confirm or change the marks, print the total. `--show` only prints |
| `utra-track leaderboard --milestone N` | Rank all runs by speed, write the speed marks into their scoresheets and print totals. `--dry-run` |
| `utra-track cameras` | Probe camera indices 0 to 9 and print the ones that open with their resolution |
| `utra-track check` | Print Python, torch, CUDA, GPU, whether CUDA works, Ultralytics version and whether the model file exists |

`run` also takes `--milestone 2|3|0`, `--team`, `--run`, `--collisions` / `--no-collisions` and `--no-marks`.

Press `q` or `Esc` in the window to stop, or Ctrl+C in the terminal. Both close the camera and flush the CSV.

## Grading a run

The overhead camera must see the whole 8 ft by 4 ft maze.

1. **Calibrate once per camera mount.** Run `utra-track calibrate`, then click the four inside floor corners of the arena in this order:
   1. the loading zone corner
   2. the far corner along the 8 ft side
   3. the corner diagonal from the loading zone
   4. the last corner
   
   The maze walls are drawn over the image so you can check them, and Enter saves `config/arena_calibration.json`. Without calibration the whole frame is taken as the maze and the dashboard shows a red warning.
2. **Start the dashboard:** `python start.py --milestone 3 --team 7 --run 1`.
3. Place the rover. The clock starts when the rover moves 0.25 ft, or when you press space. Its start point is marked **A**.
4. While the rover drives, the dashboard shows:
   - its outline and trail on the camera view and on a top down map
   - the timer against the rubric limit, 5:00 for M2 and 8:00 for M3
   - distance, 1 ft steps, turns, when it reached the loading zone, and contacts
   - the suggested marks so far
5. Press space or `q` to end the run. The end point is where the rover stopped. The terminal prints the scoresheet and offers to take the judge marks right away. Enter accepts each suggestion.
6. After all teams have run, `utra-track leaderboard --milestone 3` ranks speed and fills in the speed marks.

| Key | Action |
| --- | --- |
| space | Start or stop the run |
| `c` | Turn camera wall collision tracking on or off |
| `m` / `n` | Judge logs a minor / major collision, useful when tracking is off |
| `a` | Log an assistance (someone touched the rover) |
| `u` | Undo the last logged collision or assistance |
| `b` | Cycle the target drop off: B1, B2, B3, B4, none. B cells are numbered left to right, top to bottom from the loading zone side |
| `r` | Reset the run |
| `h` | Show or hide the key help |
| `q` / Esc | Finish and save |

**What the camera marks.** These are suggestions, and a judge mark always wins:

| Item | How it is suggested |
| --- | --- |
| Functional, drives and turns | Full marks if it drove at least 1 ft and turned at least 45°. Half if it only drove |
| No wall contact | From the minor, major and assistance tally, using the rubric steps: M3 gives 8, 7, 6 or 5, then 5 down to 2. M2 gives 2, 1.5 or 1, then 1 down to 0 |
| Arrives at loading zone (M3) | 3 if it entered an L cell after starting outside the zone. Check the LED or screen confirmation yourself |
| Finishes in B (M3) | 4 if at least half the rover outline is in the target B, 2 if it is partly inside |
| Continuous run (M3) | 5 if there were no assists and it reached the loading zone and B |
| Speed | Ranked across runs by `leaderboard`, using 1 ft steps per minute. M2 runs inside 5 minutes keep at least 2 points |

The camera cannot judge build quality, localization difficulty, pick and place, bonus, or USB tether. The judge enters those in `utra-track grade`. For M3, setting control to `line_of_sight` scales the total to 70%.

**Wall collisions.** The rover outline comes from the segmentation mask, or from the box with `grading.footprint: box`. It is mapped to maze feet and checked against the block faces and outer walls:

- A contact counts after `contact_frames` frames deeper than `contact_margin_ft`.
- It ends after `release_frames` clear frames.
- It is **major** if it goes `major_depth_ft` deep or lasts `major_duration_s`. Otherwise it is **minor**.

Tracking can be turned off in config, with `--no-collisions`, or live with `c`. The judge can then tally contacts with `m` and `n`. The camera sees the rover's top plate, which looks shifted outward near the edges of the image. If you get false contacts near the outer walls, set `arena.camera_height_ft` to correct for this, or raise `contact_margin_ft`.

Each graded run also writes these files to its folder:

- `scoresheet.yaml`: metrics, every contact with its time and place, suggested and judge marks, totals
- `track.csv`: position in feet for every frame
- `path.png`: a top down map of the path with start, end and contacts marked. It shows no people, so it is safe to keep as evidence

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
| `arena.layout` | Rubric maze | One string per row of 1 ft cells: `.` open, `#` block, `L` loading zone, `B` drop off |
| `arena.calibration` | `config/arena_calibration.json` | Written by `calibrate` |
| `arena.camera_height_ft` / `robot_height_ft` | `null` / `0.5` | Parallax correction, off while the camera height is null |
| `grading.milestone` | `3` | `2` or `3` picks the rubric, `0` shows the plain tracker |
| `grading.collisions` | `true` | Camera wall contact tracking |
| `grading.footprint` | `mask` | `mask` or `box` outline for contact checks |
| `grading.contact_margin_ft` | `0.04` | Contacts shallower than this are ignored |
| `grading.major_depth_ft` / `major_duration_s` | `0.15` / `1.0` | When a contact counts as major |
| `grading.contact_frames` / `release_frames` | `3` / `6` | Debounce |
| `grading.auto_start` / `start_move_ft` | `true` / `0.25` | Start the clock on first movement |
| `grading.time_limit_s` | `null` | Uses the rubric limit: 300 s for M2, 480 s for M3 |

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
3. Check the contact thresholds against judge tallies from real runs.
