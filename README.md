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
| 2 | Ink Tide (`ink-tide`) | Kept | 2-D incompressible Navier–Stokes (stable fluids on the GPU); every force is a vortex pair |
| 3 | Ripple Tank (`ripple-tank`) | On trial | 2-D damped wave equation; 12 note sources at pitch-scaled frequencies interfere, a refracting lens focuses caustics |
| 4 | Flux Lattice (`flux-lattice`) | On trial | Wave equation on a flexible 2-D grid that kicks bend and claps snap; each melody note releases a light pulse that bounces off the frame and is thrown by the grid's slope; cymbal lasers reflect round the frame |
| 5 | Deep Voyage (`deep-voyage`) | On trial | 3-D perspective flight; speed from the story energy; noise-tunnel gas clouds; shaded planets with rings; black holes with lens-equation light bending and accretion discs |
| 6 | Signal Plane (`signal-plane`) | On trial | Perspective data plane: instanced glyph streams race along 12 cables (one per note), split and reverse on claps; ray-cast plane shader for grid, cables, scan lines and laser-sheet smoke; wireframe globes, ringed planets, moons and gravity wells pass beneath |
| 7 | Wire City (`wire-city`) | On trial | Procedural wireframe city between three planes (street dot grid, rippling point-cloud sky membrane, sweeping scan planes); melody packets beam rooftop → membrane → across as characters → down into another tower; camera modes per variation |
| 8 | Data Forest (`data-forest`) | On trial | Procedural branching wireframe trees bent by a rolling wind field (height-weighted sway); glyph data rain with ground splashes; leaves that tear loose on claps; gentle seahorse-like light beings drifting slowly on the trunks, swaying in the wind; camera views per variation |
| 9 | Aurora Shelf (`aurora-shelf`) | On trial | Polar night: wireframe ice plateau and cliff, swell-line sea, bobbing icebergs, aurora curtains lit by melody, data snow, kick cracks that fork across the ice, wireframe whales (a motion slider up to breaching) that blow spouts on claps, schooling fish shoals, and streams of characters that streak at the top of their slider |

### Story

The **Story** tab (next to the visualiser menu) holds the song's arc: chapters (analysis sections,
long ones split into ~20 s pieces snapped to kicks), each with an energy level, a shape (Hold,
Build, Fade out, Follow music), which effects are on, and a variation. Sections that sound alike
share a letter (A1, A2…). The default story builds up to the drop (the first full-energy chapter
after a breakdown), adding one effect at a time, then follows the music. The energy line brightens
every visualiser; #4 also uses the effects and variations. Edits are saved in the browser per track.

To remove a visualiser, delete its entry from `VISUALISERS` and its class/config block.

Open `web/index.html` in Chrome, Safari or Firefox (desktop or phone). It starts with the demo
track `web/Mix.mp3` (falls back to a built-in synth beat if the file cannot be fetched, e.g. when
the page is opened straight from disk); use **Load MP3 or WAV** for any other track.

- **Analysis** is a JavaScript port of `analyse.py` (bands, kick/snare/hat hits, chroma,
  brightness, loudness, song sections). On a test track it finds the same kick and hat hits as
  the Python version, and 79 of Python's 83 snare hits.
- **Physics** is a line-for-line port of `render.py` (same CONFIG names and values, same seed).
- **Rendering** uses WebGL2 with float buffers, matching the Python brightness at both render sizes.
- **Format**: Vertical 9:16 (1080×1920), Square 1:1 (1080×1080) or Landscape 16:9 (1920×1080), chosen in the Export panel before recording. Simulation grids resize to keep square cells.
- **Recording** uses the browser's MediaRecorder: it plays the track once, in real time.
  Chrome and Safari record MP4 (H.264); Firefox records WebM.

## Python version

```bash
pip install librosa numba opencv-python scipy numpy   # plus ffmpeg
prototypes/gravity_harp/make_video.sh track.mp3 out.mp4
CRF=25 CHUNKS=4 prototypes/gravity_harp/make_video.sh track.mp3 out.mp4   # smaller file, 4 cores
python3 prototypes/gravity_harp/render.py features.npz --stills 30,90     # preview PNGs
```
