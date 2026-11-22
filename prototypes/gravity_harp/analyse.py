"""
analyse.py — turn an audio file into per-video-frame feature curves.

Usage:  python3 analyse.py track.mp3 features.npz [--fps 30]

Every feature is sampled once per video frame and normalised to 0..1,
so the renderer never has to think about audio.

Features
  sub, bass, lowmid, mid, presence, air   band energies (smoothed)
  kick, snare, hat                        onset envelopes (spiky, decay)
  kick_hits, snare_hits, hat_hits         boolean frame indices of hits
  chroma (N,12)                           pitch-class strengths (harmonic part)
  tonal                                   how "melodic" vs "noisy" the frame is
  rms                                     loudness
  centroid                                spectral brightness
  section (N,) int, section_energy        song structure (agglomerative segments)
"""
import sys, argparse
import numpy as np
import librosa
from scipy.ndimage import uniform_filter1d, maximum_filter1d

ap = argparse.ArgumentParser()
ap.add_argument("audio")
ap.add_argument("out")
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--sections", type=int, default=8)
a = ap.parse_args()

SR = 22050
HOP = 512
y, sr = librosa.load(a.audio, sr=SR, mono=True)
dur = len(y) / sr
nframes = int(np.ceil(dur * a.fps))
print(f"duration {dur:.2f}s -> {nframes} video frames")

y_h, y_p = librosa.effects.hpss(y, margin=2.0)

S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
Sp = np.abs(librosa.stft(y_p, n_fft=2048, hop_length=HOP))
freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
t_audio = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP)
t_video = np.arange(nframes) / a.fps


def band(spec, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return np.log1p(spec[m].mean(axis=0) * 50)


def to_video(x):
    return np.interp(t_video, t_audio, x)


def norm(x, lo_pct=5, hi_pct=99):
    lo, hi = np.percentile(x, lo_pct), np.percentile(x, hi_pct)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def smooth_attack_release(x, attack=0.6, release=0.12):
    out = np.zeros_like(x)
    v = 0.0
    for i, s in enumerate(x):
        k = attack if s > v else release
        v += (s - v) * k
        out[i] = v
    return out


feats = {}
bands = {"sub": (20, 60), "bass": (60, 160), "lowmid": (160, 500),
         "mid": (500, 2000), "presence": (2000, 6000), "air": (6000, 11000)}
for k, (lo, hi) in bands.items():
    feats[k] = smooth_attack_release(norm(to_video(band(S, lo, hi))))


def onset_env(lo, hi, spec):
    m = (freqs >= lo) & (freqs < hi)
    env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(spec[m]), sr=sr,
                                       hop_length=HOP)
    return env


def hits(env, delta, wait_s):
    idx = librosa.util.peak_pick(env, pre_max=3, post_max=3, pre_avg=10,
                                 post_avg=10, delta=delta,
                                 wait=int(wait_s * sr / HOP))
    tv = np.round(t_audio[idx] * a.fps).astype(int)
    ok = (tv >= 0) & (tv < nframes)          # keep hit times and strengths aligned
    strength = norm(env)[idx][ok]
    return tv[ok], strength


def decay_env(hit_frames, strength, decay):
    e = np.zeros(nframes)
    e[hit_frames] = np.maximum(e[hit_frames], strength)
    out = np.zeros(nframes)
    v = 0
    for i in range(nframes):
        v = max(v * decay, e[i])
        out[i] = v
    return out


kick_env = onset_env(30, 150, S)
snare_env = onset_env(1500, 5000, Sp)
hat_env = onset_env(7000, 11000, Sp)

for name, env, delta, wait, dec in [("kick", kick_env, 0.35, 0.18, 0.80),
                                    ("snare", snare_env, 0.40, 0.20, 0.82),
                                    ("hat", hat_env, 0.30, 0.08, 0.70)]:
    envn = env / (np.percentile(env, 99) + 1e-9)
    hf, st = hits(envn, delta, wait)
    feats[name + "_hits"] = hf
    feats[name + "_hit_strength"] = st
    feats[name] = decay_env(hf, np.clip(st, 0.3, 1), dec)
    print(f"{name}: {len(hf)} hits")

chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=HOP)
chroma_v = np.stack([to_video(c) for c in chroma], axis=1)
chroma_v = uniform_filter1d(chroma_v, size=5, axis=0)
chroma_v /= chroma_v.max(axis=1, keepdims=True) + 1e-9
feats["chroma"] = chroma_v

flat = librosa.feature.spectral_flatness(y=y, hop_length=HOP)[0]
feats["tonal"] = smooth_attack_release(1 - norm(to_video(flat)), 0.2, 0.08)

h_rms = librosa.feature.rms(y=y_h, hop_length=HOP)[0]
feats["harmonic"] = smooth_attack_release(norm(to_video(h_rms)), 0.3, 0.08)

rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
feats["rms"] = smooth_attack_release(norm(to_video(rms)), 0.4, 0.06)

cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP)[0]
feats["centroid"] = uniform_filter1d(norm(to_video(cent)), 9)

# --- structure ---
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
sync = np.vstack([librosa.util.normalize(mfcc, axis=1),
                  librosa.util.normalize(chroma, axis=1)])
beat_hop = int(2.0 * sr / HOP)  # ~2 s blocks for robust segmentation
blocks = librosa.util.sync(sync, np.arange(0, sync.shape[1], beat_hop))
# short clips have fewer ~2 s blocks than requested sections
n_sec = int(max(1, min(a.sections, blocks.shape[1])))
if n_sec < a.sections:
    print(f"short clip: using {n_sec} sections instead of {a.sections}")
a.sections = n_sec
bounds = librosa.segment.agglomerative(blocks, a.sections)
bound_t = bounds * beat_hop * HOP / sr
sec = np.searchsorted(bound_t, t_video, side="right") - 1
feats["section"] = np.clip(sec, 0, a.sections - 1)
sec_e = np.array([feats["rms"][feats["section"] == s].mean()
                  if np.any(feats["section"] == s) else 0
                  for s in range(a.sections)])
feats["section_energy_table"] = sec_e
long_rms = uniform_filter1d(feats["rms"], size=a.fps * 4)
feats["energy_slow"] = norm(long_rms, 1, 99)
print("sections at", np.round(bound_t, 1), "energy", np.round(sec_e, 2))

feats["fps"] = a.fps
feats["duration"] = dur
feats["nframes"] = nframes
np.savez_compressed(a.out, **feats)
print("saved", a.out)
