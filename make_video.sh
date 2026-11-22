#!/usr/bin/env bash
# Usage: ./make_video.sh track.mp3 [output.mp4]
# Analyses the track, renders the visual in two parallel chunks,
# joins them losslessly and muxes the original audio.
set -euo pipefail
AUDIO="$1"
OUT="${2:-visualiser.mp4}"
cd "$(dirname "$0")"

python3 analyse.py "$AUDIO" features.npz
DUR=$(python3 -c "import numpy as np; print(float(np.load('features.npz')['duration']))")
MID=$(python3 -c "print(round($DUR/2))")

python3 render.py features.npz --start 0 --end "$MID" --out part1.mp4 > part1.log 2>&1 &
P1=$!
python3 render.py features.npz --start "$MID" --out part2.mp4 > part2.log 2>&1 &
P2=$!
wait $P1; wait $P2

printf "file 'part1.mp4'\nfile 'part2.mp4'\n" > parts.txt
ffmpeg -y -loglevel error -f concat -safe 0 -i parts.txt -c copy video_only.mp4
ffmpeg -y -loglevel error -i video_only.mp4 -i "$AUDIO" -map 0:v -map 1:a \
       -c:v copy -c:a aac -b:a 256k -shortest -movflags +faststart "$OUT.partial.mp4"
mv "$OUT.partial.mp4" "$OUT"
echo "wrote $OUT"
