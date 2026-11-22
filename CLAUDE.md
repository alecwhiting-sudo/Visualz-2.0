# Audio Visualiser Series — Project Brief

> Context for Claude Code. Written from the chat where the first visualiser
> ("Gravity Harp") was prototyped. Owner: Alec (DIVURJ). Alec does not write
> code by hand and directs the product, so explain trade-offs in plain language
> and keep the stack lean and cheap.

## Intent

Build a **series of audio-reactive visualisers** that:

- take any **MP3 or WAV** the user loads and generate a visualisation from it
- are **designed for mobile**: vertical **1080×1920, 30 fps** video, ready
  for Reels, TikTok and Shorts
- look **artistic and abstract**, but are **driven by real physics**
  (gravity, viscosity, wave equations, shockwaves, turbulence) rather than
  bars and waveforms
- map **each part of the music to its own kind of visual feature**, so you
  can see the kick, the snare, the bass and the harmony as separate things
- are **code-based, re-runnable and easy to edit**: each visualiser exposes a
  clear block of settings (forces, palettes, glow, trails)
- ship as a **browser app** first, and possibly an **iOS app** later

## What already exists (prototype)

Three files from the first session belong in `prototypes/gravity_harp/`:

| File | Role |
|---|---|
| `analyse.py` | Audio → per-video-frame feature curves (`features.npz`), using librosa |
| `render.py` | Physics simulation and drawing (numpy, numba, OpenCV); pipes raw frames to ffmpeg |
| `make_video.sh` | One command: analyse, render in 2 parallel chunks, join losslessly, add the audio |

Run: `./make_video.sh track.mp3 out.mp4`
Quick preview: `python3 render.py features.npz --stills 30,90` (PNGs, about 20 s)
Dependencies: `pip install librosa numba opencv-python scipy numpy` plus ffmpeg.

### Gravity Harp: how the music maps to the visuals

| Part of the music | Physics / visual |
|---|---|
| Sub + bass | Mass of the central gravity well; the photon ring swells |
| Kick | Radial pressure blast from the core (the disk "breathes") plus a small camera punch-in |
| Snare | Expanding shockwave rings that push particles |
| Low mids | Disk spin (target orbital speed) |
| Hats + air | Curl-noise turbulence plus spark emission |
| Chroma (12 pitch classes) | 12 vibrating strings (1-D wave equation), coloured on the circle of fifths |
| Spectral centroid | Warmth of the particle colours |
| Song sections | Palette changes; quiet sections loosen gravity so the disk opens into a halo |

### Lessons from tuning (keep these for future visualisers)

- **Feature extraction is shared and reusable.** Every visualiser should consume
  the same normalised 0–1 per-frame features, so the analysis code doesn't
  need to know anything about the visuals.
- **Energy blow-up is the main physics risk.** Repeated impulses (kicks,
  snares, turbulence) pumped energy in until particles flew off. What fixed it:
  viscosity that pulls each particle back towards its own *home orbit*, not a
  shared radius (a shared radius collapsed everything into a thin ring).
  Impulses applied every frame over a band, like the snare rings, must be
  scaled small.
- Always check **still frames at the busiest moments** before a full render.
- **Rendering cost:** about 7 frames per second on 2 CPU cores, so a 3-minute
  track takes about 10 minutes split across 2 chunks. The simulation is cheap
  (it can replay from frame 0); drawing and encoding are the expensive part.
- **File size:** fine particle grain and noise make the encode heavy (593 MB at
  CRF 19, 150 MB at CRF 25). Add export presets for sharing.

## Goals for the repo

1. **A shared core**: audio loading (MP3/WAV), feature analysis, render loop,
   ffmpeg export, presets.
2. **Pluggable visualisers**: each one is a module with the same interface
   (`Scene(features)`, `step(i)`, `draw(i) -> frame`, plus a `CONFIG` block).
   Gravity Harp is visualiser #1.
3. **A browser app (mobile-first)**: upload a track, pick a visualiser, tweak
   a few settings, preview, export or download a vertical video.
4. **iOS app later**, possibly.

## Open decisions (ask Alec before committing)

- **Where does rendering happen?** This decides the architecture:
  - **Server-side Python render** (keeps these scripts as they are): the browser
    uploads the track and gets back an MP4. This needs a worker that can run
    for minutes (Vercel serverless functions can't; it needs a container or
    job host), and cost grows with use.
  - **In-browser real-time** (WebGL/WebGPU in JS or TS): Python stays as the
    reference implementation, or features are pre-computed, and the browser
    renders live and records video. No server cost, works on iOS through a
    wrapper or a native Metal port, but the visual code has to be ported.
  - **Hybrid**: analysis in Python or WASM, rendering in the browser.
- How much users can customise (presets only, or exposed sliders).
- Monetisation and licensing (Alec's existing stack: GitHub, Vercel free tier,
  Lemon Squeezy as merchant of record).

## Constraints and preferences

- Lean and low-cost. Prefer free tiers and avoid heavy infrastructure until it's needed.
- Output default: 1080×1920, 30 fps, H.264 + AAC, `+faststart`.
- Keep every visualiser deterministic (seeded) so renders are reproducible.
- Keep settings readable at the top of each visualiser file.

## Suggested layout

```
core/            audio loading, analyse.py, render loop, ffmpeg export
visualisers/
  gravity_harp/  render.py (Scene + CONFIG)
  ...            future visualisers
prototypes/gravity_harp/   original files from the first session
web/             browser app (once the render location is decided)
```
