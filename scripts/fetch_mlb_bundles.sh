#!/bin/bash
# Re-download MLB's Gameday JS bundles into samples/mlb_bundles/.
# These are PUBLIC unauthenticated static assets — no JWT, no browser
# session needed. They contain the FlatBuffer schema classes we need
# for decoding the .bin tracking data (TrackingDataWire,
# TrackingFrameWire, ActorPoseWire, SkeletalPlayerWire,
# InferredBat/TrackingBatPositionWire, plus all the gameEvents and
# trackedEvents schemas that Phase A of PROJECT_PREDICTION.md needs).
#
# Run from anywhere; writes to ~/fieldvision/samples/mlb_bundles/.
set -e
DEST="${HOME}/fieldvision/samples/mlb_bundles"
BASE="https://prod-gameday.mlbstatic.com/app-mlb/5.50.0-mlb.5"
mkdir -p "$DEST"
for f in gd.min.js gd.@bvg_poser.min.js gd.@bvg_poser-fallback.min.js; do
  echo -n "  $f ... "
  curl -sS -o "$DEST/$f" "$BASE/$f"
  size=$(wc -c < "$DEST/$f")
  echo "$size bytes"
done
echo "saved to $DEST"
