# ThrowTrack

Counting and characterizing pitchers' **non-mound throws** (catch play, long
toss, plyo wall drills) from ordinary video: who threw, to whom, when, which
ball, how far, how hard, and how sure the system is about each of those.

Built as a prototype for the Boston Red Sox Innovation Challenge (Pitcher
Workload). **Status: early prototype, tested on simulated sessions only.** No
real challenge data has been released yet, and the detection weights and
thresholds are placeholders meant to be re-fit on labeled sessions.

## How it works

```
video ──► person detection + tracking (YOLO, BoT-SORT) ──► 2D pose per player (RTMPose-X)
                                     │   cached once per video (poses.npz)
                                     ▼
   throws.py    isolated bursts of throwing-hand speed, in body lengths/s → P(throw)
   pairing.py   partners alternate and throw toward each other → receiver; else wall drill
   geometry.py  feet → ground plane (4 cones once per camera spot, or roster heights)
   ball_type.py ball color in the throwing hand before release → staff-editable palette
   velocity.py  drag-corrected flight physics; hand-speed regression calibrated on radar
   report.py    throws.csv / players.csv / report.html, with per-throw confidence + review flags
```

Only `video.py` needs the GPU stack. Everything downstream works on pose
tracks, so the stages can be re-run and re-tuned in seconds without
reprocessing video.

## Try it (no GPU, no video)

```bash
cd throwtrack
PYTHONPATH=src python -m throwtrack demo --out runs/demo
```

This simulates a 6-minute session (three throwing pairs, including one that
joins late, one that backs up for long toss and one at an oblique angle, plus
a wall-drill player with plyo balls). It runs the full pipeline and scores it
against ground truth. Typical output:

| Stage | Simulated result |
|---|---|
| Throw detection | precision 1.00, recall 0.99, release timing sd ≈ 25 ms |
| Receiver / wall-drill labeling | 100% |
| Partner distance (cone calibration) | MAE ≈ 3 ft at 60–100 ft |
| Velocity, hand-speed model only (no ball tracking) | MAE ≈ 8–10 mph on held-out throws |

The simulator checks that the logic is right end to end. It says nothing
about real-world accuracy. Velocity from 30 fps pose alone is the weakest
stage, which is why ball flight timing is the main velocity method in the
plan and the pose model is the fallback.

## On real video

```bash
PYTHONPATH=src python -m throwtrack analyze session.mp4 --out runs/s1 \
    --calibration calib.json --roster roster.json --ball-type --overlay
PYTHONPATH=src python -m throwtrack fit-velocity --features runs/s1/throw_features.json \
    --radar radar.csv --out velocity_model.json
```

`calibration.json` is either
`{"mode": "homography", "image_points": [[x, y], ...], "ground_points_ft": [[X, Y], ...]}`
(at least 4 cones or markings) or
`{"mode": "person_height", "image_size": [1920, 1080], "hfov_deg": 80}`.
`roster.json` maps track ids to `{"name": ..., "height_in": ...}`.

The video layer (`extract_pose_tracks`) has not yet been run against real
footage or model weights. The OpenCV parts (ball-pixel extraction, overlay
rendering) are tested on a synthetic rendered clip.

## Tests

```bash
cd throwtrack && python -m pytest tests -q
```

## Next, once data arrives

1. Audit the footage (camera angles, frame rate, resolution, occlusion), then decide which signals lead.
2. Label throws in a few sessions and re-fit the throw scorer (`ThrowScorer.fit`).
3. Add ball detection and trajectory fitting for sub-frame flight time, and use it as the primary velocity source.
4. Replace color-voting ball typing with a small CNN on hand crops if colors are ambiguous.
5. Build a simple staff UI: upload, confirm player names, review flagged throws.
