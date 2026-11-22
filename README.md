# Visualz

Physics-driven, audio-reactive visualisers for vertical video (1080×1920, 30 fps).
See `CLAUDE.md` for the full brief.

## What's here

| Path | What it is |
|---|---|
| `web/index.html` | **Browser proof of concept** of Gravity Harp. One self-contained page: load an MP3/WAV, it analyses the track in the browser, runs the physics live (WebGL2) and records a vertical video. No server. |
| `prototypes/gravity_harp/` | The original Python version (reference implementation): `analyse.py`, `render.py`, `make_video.sh`. |

## Web POC

The page has a menu of visualisers. Each one shares the audio analysis, playback, recording and
glow pass, and brings its own physics, settings and palette (see `VISUALISERS` in `web/index.html`).
A link ending in `#ink-tide` (or any visualiser id) opens that visualiser directly.

| # | Visualiser | Status | Physics |
|---|---|---|---|
| 1 | Gravity Harp (`gravity-harp`) | Kept | Particles under gravity + viscosity, 1-D wave-equation strings, shockwaves |
| 2 | Ink Tide (`ink-tide`) | On trial | 2-D incompressible Navier–Stokes (stable fluids on the GPU); every force is a vortex pair |

To remove a visualiser, delete its entry from `VISUALISERS` and its class/config block.

Open `web/index.html` in Chrome, Safari or Firefox (desktop or phone). It starts with the demo
track `web/Mix.mp3` (falls back to a built-in synth beat if the file cannot be fetched, e.g. when
the page is opened straight from disk); use **Load MP3 or WAV** for any other track.

- **Analysis** is a JavaScript port of `analyse.py` (bands, kick/snare/hat hits, chroma,
  brightness, loudness, song sections). On a test track it finds the same kick and hat hits as
  the Python version, and 79 of Python's 83 snare hits.
- **Physics** is a line-for-line port of `render.py` (same CONFIG names and values, same seed).
- **Rendering** uses WebGL2 with float buffers, matching the Python brightness at both render sizes.
- **Recording** uses the browser's MediaRecorder: it plays the track once, in real time.
  Chrome and Safari record MP4 (H.264); Firefox records WebM.

## Python version

```bash
pip install librosa numba opencv-python scipy numpy   # plus ffmpeg
prototypes/gravity_harp/make_video.sh track.mp3 out.mp4
CRF=25 CHUNKS=4 prototypes/gravity_harp/make_video.sh track.mp3 out.mp4   # smaller file, 4 cores
python3 prototypes/gravity_harp/render.py features.npz --stills 30,90     # preview PNGs
```
