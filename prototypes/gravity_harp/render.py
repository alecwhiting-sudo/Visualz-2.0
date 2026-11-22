"""
render.py — "Gravity Harp": a physics-driven audio visualiser (1080x1920, 30 fps).

Pipeline:   python3 analyse.py track.mp3 features.npz
            python3 render.py features.npz --out part.mp4 [--start S --end E]
            python3 render.py features.npz --stills 12,45,90     (preview PNGs)
            ./make_video.sh track.mp3                             (whole thing)

What each element of the music drives
  SUB + BASS   -> mass of the central gravity well (tighter, faster orbits)
  KICK         -> radial pressure blast from the core (the disk "breathes")
                  + a small camera punch-in
  SNARE        -> shockwave rings that travel outward and shove particles
  LOW-MIDS     -> vortex torque (spins the accretion disk)
  HATS + AIR   -> curl-noise turbulence and spark emission
  CHROMA       -> 12 vibrating strings (one per pitch class, coloured on the
                  circle of fifths) solved with the 1-D wave equation
  BRIGHTNESS   -> warmth of the particle colours
  SECTIONS     -> palette changes; quiet sections loosen gravity so the disk
                  relaxes into a halo
Everything below CONFIG is plain numpy/numba — edit freely.
"""
import argparse, os, subprocess, sys, math
import numpy as np
import cv2
from numba import njit

# ============================ CONFIG ======================================
W, H = 1080, 1920
SEED = 7
N_PARTICLES = 34000
N_SPARKS = 4000
STRING_NODES = 72
STRING_INNER, STRING_OUTER = 150.0, 820.0      # px from centre
CENTER = (W * 0.5, H * 0.47)

G_BASE = 1.6e7          # gravity strength (px^3/s^2); multiplied by bass
G_BASS = 2.2e7
SOFTEN = 60.0           # gravitational softening radius
KICK_PUSH = 280.0       # px/s radial impulse on kick
KICK_RANGE = 380.0      # e-folding radius of kick impulse
SNARE_SPEED = 950.0     # shockwave speed px/s
SNARE_PUSH = 40.0 
SNARE_WIDTH = 70.0
TORQUE = 0.25           # low-mid: extra orbital spin (fraction)
TURB = 55.0             # hat/air turbulence strength
DISK_INNER = 90.0       # disk inner edge, px
DISK_SPREAD = 175.0     # disk radial scale, px
DISK_RESTORE = 2.0      # how hard matter drifts back toward it, 1/s
DRAG = 1.6              # viscosity: relaxation rate toward orbit, 1/s
TRAIL = 0.80            # framebuffer persistence per frame (0..1)
EXPOSURE = 1.6
GLOW = 1.1
PARTICLE_INT = 0.15
STRING_INT = 1.6
SPIN_BASE, SPIN_ENERGY = 0.03, 0.12   # camera rotation rad/s

# Section palettes: (background, cold, hot, accent)  — linear RGB 0..1
PALETTES = [
    ((0.010, 0.012, 0.035), (0.10, 0.35, 1.00), (1.00, 0.45, 0.85), (0.6, 0.9, 1.0)),
    ((0.020, 0.008, 0.030), (0.55, 0.15, 1.00), (1.00, 0.60, 0.25), (1.0, 0.8, 0.6)),
    ((0.004, 0.018, 0.025), (0.05, 0.70, 0.75), (0.75, 1.00, 0.85), (0.6, 1.0, 0.9)),
    ((0.010, 0.010, 0.030), (0.25, 0.30, 1.00), (0.95, 0.95, 1.00), (0.8, 0.8, 1.0)),
    ((0.025, 0.006, 0.015), (1.00, 0.18, 0.35), (1.00, 0.85, 0.40), (1.0, 0.6, 0.5)),
    ((0.008, 0.014, 0.035), (0.15, 0.55, 1.00), (1.00, 0.35, 0.60), (0.7, 0.9, 1.0)),
    ((0.018, 0.008, 0.032), (0.65, 0.20, 1.00), (0.30, 1.00, 0.90), (0.9, 0.7, 1.0)),
    ((0.006, 0.006, 0.020), (0.30, 0.40, 0.80), (0.80, 0.70, 1.00), (0.7, 0.7, 0.9)),
]
FADE_IN, FADE_OUT = 1.5, 3.0
# ==========================================================================

FPS = 30            # overwritten from features.npz in main(), so analysis and
DT = 1.0 / FPS      # render always agree on the frame rate


def hsv(h, s, v):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


# pitch class -> hue on the circle of fifths, so consonant notes sit nearby
STRING_COLORS = np.array([hsv(((k * 7) % 12) / 12.0, 0.65, 1.0) for k in range(12)],
                         dtype=np.float32)


# ----------------------------- physics kernels ----------------------------
@njit(cache=True, fastmath=True)
def curl(x, y, t):
    # analytic curl of a sum-of-sines stream function -> divergence-free flow
    k1, k2, k3 = 0.0045, 0.0081, 0.0137
    vx = (math.cos(k1 * y + 0.7 * t) * k1 * 1.0 +
          math.cos(k2 * (x + y) * 0.7 - 1.1 * t) * k2 * 0.7 * 0.6 +
          math.cos(k3 * y - k3 * 0.5 * x + 1.9 * t) * k3 * 0.35)
    vy = -(math.cos(k1 * x - 0.5 * t) * k1 * 1.0 +
           math.cos(k2 * (x + y) * 0.7 - 1.1 * t) * k2 * 0.7 * 0.6 +
           math.cos(k3 * y - k3 * 0.5 * x + 1.9 * t) * (-k3 * 0.5) * 0.35)
    return vx / 0.0045, vy / 0.0045


@njit(cache=True, fastmath=True)
def step_particles(px, py, vx, vy, cx, cy, gm, kick_imp, torque, turb, drag,
                   rings, nrings, t, dt, rnd, home, disk_scale):
    n = px.shape[0]
    for i in range(n):
        dx = px[i] - cx
        dy = py[i] - cy
        r2 = dx * dx + dy * dy
        r = math.sqrt(r2) + 1e-3
        s2 = r2 + SOFTEN * SOFTEN
        inv = 1.0 / (s2 * math.sqrt(s2))
        ax = -gm * dx * inv
        ay = -gm * dy * inv
        # viscosity: gas-like relaxation toward a circular (Keplerian) orbit.
        # 'torque' (low-mids) speeds the target orbit up -> the disk spins.
        vc = math.sqrt(gm * r2 * inv) * torque
        # disk pressure: drift back toward the preferred disk radius
        vr = -DISK_RESTORE * (r - home[i] * disk_scale)
        ax += -drag * (vx[i] - (-dy / r) * vc - dx / r * vr)
        ay += -drag * (vy[i] - (dx / r) * vc - dy / r * vr)
        # turbulence
        tx, ty = curl(px[i], py[i], t)
        ax += tx * turb
        ay += ty * turb
        vxi = vx[i] + ax * dt
        vyi = vy[i] + ay * dt
        # kick: radial pressure blast
        if kick_imp > 0.0:
            k = kick_imp * math.exp(-r / KICK_RANGE)
            vxi += dx / r * k
            vyi += dy / r * k
        # snare shockwaves
        for j in range(nrings):
            rx = px[i] - rings[j, 0]
            ry = py[i] - rings[j, 1]
            d = math.sqrt(rx * rx + ry * ry) + 1e-3
            off = d - rings[j, 2]
            if -SNARE_WIDTH < off < SNARE_WIDTH * 0.3:
                f = rings[j, 3] * (1.0 - abs(off) / SNARE_WIDTH)
                vxi += rx / d * f
                vyi += ry / d * f
        px[i] += vxi * dt
        py[i] += vyi * dt
        vx[i] = vxi
        vy[i] = vyi
        # respawn if swallowed or escaped
        if r < 22.0 or r > 1350.0:
            a = rnd[i, 0] * 6.2831853
            rr = home[i] * disk_scale
            px[i] = cx + math.cos(a) * rr
            py[i] = cy + math.sin(a) * rr
            s2 = rr * rr + SOFTEN * SOFTEN
            vc = math.sqrt(gm * rr * rr / (s2 * math.sqrt(s2)))
            vx[i] = -math.sin(a) * vc
            vy[i] = math.cos(a) * vc


@njit(cache=True, fastmath=True)
def splat(buf, xs, ys, cols, ints, cx, cy, ca, sa, zoom):
    h, w = buf.shape[0], buf.shape[1]
    for i in range(xs.shape[0]):
        dx = (xs[i] - cx) * zoom
        dy = (ys[i] - cy) * zoom
        x = cx + dx * ca - dy * sa
        y = cy + dx * sa + dy * ca
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        if x0 < 0 or y0 < 0 or x0 >= w - 1 or y0 >= h - 1:
            continue
        fx = x - x0
        fy = y - y0
        it = ints[i]
        for c in range(3):
            v = cols[i, c] * it
            buf[y0, x0, c] += v * (1 - fx) * (1 - fy)
            buf[y0, x0 + 1, c] += v * fx * (1 - fy)
            buf[y0 + 1, x0, c] += v * (1 - fx) * fy
            buf[y0 + 1, x0 + 1, c] += v * fx * fy


@njit(cache=True, fastmath=True)
def particle_colors(vx, vy, cold, hot, warmth, out_col, out_int, base):
    for i in range(vx.shape[0]):
        s = math.sqrt(vx[i] * vx[i] + vy[i] * vy[i])
        t = min(1.0, s / (650.0 - 250.0 * warmth))
        t = t * t * (3 - 2 * t)
        for c in range(3):
            out_col[i, c] = cold[c] * (1 - t) + hot[c] * t
        out_int[i] = base * (0.45 + 1.3 * t)


@njit(cache=True, fastmath=True)
def step_strings(u, v, drive_amp, drive_harm, t, dt, substeps):
    ns, nn = u.shape
    c2 = 55.0 * 55.0
    h = dt / substeps
    for _ in range(substeps):
        for k in range(ns):
            f = drive_amp[k] * math.sin(6.2831853 * 0.45 * drive_harm[k] * t + k)
            for n in range(1, nn - 1):
                lap = u[k, n - 1] - 2 * u[k, n] + u[k, n + 1]
                a = c2 * lap - 2.2 * v[k, n]
                if n == 6:
                    a += f
                v[k, n] += a * h
            for n in range(1, nn - 1):
                u[k, n] += v[k, n] * h
        t += h


@njit(cache=True, fastmath=True)
def step_sparks(sx, sy, svx, svy, life, cx, cy, gm, turb, t, dt):
    for i in range(sx.shape[0]):
        if life[i] <= 0:
            continue
        dx = sx[i] - cx
        dy = sy[i] - cy
        r2 = dx * dx + dy * dy + SOFTEN * SOFTEN
        inv = 1.0 / (r2 * math.sqrt(r2))
        tx, ty = curl(sx[i] * 1.7, sy[i] * 1.7, t * 1.5)
        svx[i] += (-gm * dx * inv * 0.5 + tx * turb * 2.0) * dt
        svy[i] += (-gm * dy * inv * 0.5 + ty * turb * 2.0) * dt
        svx[i] *= 1 - 1.6 * dt
        svy[i] *= 1 - 1.6 * dt
        sx[i] += svx[i] * dt
        sy[i] += svy[i] * dt
        life[i] -= dt


# ------------------------------- the scene --------------------------------
class Scene:
    def __init__(self, F):
        self.F = F
        rng = np.random.default_rng(SEED)
        self.rng = rng
        cx, cy = CENTER
        a = rng.uniform(0, 2 * np.pi, N_PARTICLES)
        # each particle's home orbit: dense inner disk, long faint tail
        r = DISK_INNER + rng.gamma(1.8, DISK_SPREAD, N_PARTICLES)
        r = np.minimum(r, 1150)
        self.home = r.copy()
        self.px = (cx + np.cos(a) * r).astype(np.float64)
        self.py = (cy + np.sin(a) * r).astype(np.float64)
        gm = G_BASE
        s2 = r * r + SOFTEN ** 2
        vc = np.sqrt(gm * r * r / s2 ** 1.5)
        self.vx = -np.sin(a) * vc
        self.vy = np.cos(a) * vc
        self.pcol = np.zeros((N_PARTICLES, 3), np.float32)
        self.pint = np.zeros(N_PARTICLES, np.float32)

        self.u = np.zeros((12, STRING_NODES))
        self.uv = np.zeros((12, STRING_NODES))
        self.harm = np.array([1 + (k * 5) % 4 for k in range(12)], float)
        self.prev_chroma = np.zeros(12)

        self.sx = np.zeros(N_SPARKS); self.sy = np.zeros(N_SPARKS)
        self.svx = np.zeros(N_SPARKS); self.svy = np.zeros(N_SPARKS)
        self.slife = np.zeros(N_SPARKS); self.smax = np.ones(N_SPARKS)
        self.scol = np.zeros((N_SPARKS, 3), np.float32)
        self.spark_ptr = 0

        self.rings = np.zeros((16, 5))   # cx, cy, radius, push, age
        self.nrings = 0
        self.ring_side = 0

        self.theta = 0.0
        self.pal = np.array(PALETTES[0], float)
        self.buf = np.zeros((H, W, 3), np.float32)
        self.kick_set = set(F["kick_hits"].tolist())
        self.kick_str = dict(zip(F["kick_hits"].tolist(), F["kick_hit_strength"].tolist()))
        self.snare_set = dict(zip(F["snare_hits"].tolist(), F["snare_hit_strength"].tolist()))
        self.hat_set = dict(zip(F["hat_hits"].tolist(), F["hat_hit_strength"].tolist()))
        self.sec_energy = F["section_energy_table"]

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        d = np.sqrt((xx - cx) ** 2 + ((yy - cy) * 0.8) ** 2) / 1100.0
        self.bg_shape = np.clip(1.0 - d, 0, 1)[..., None] ** 1.6
        self.vignette = np.clip(1.15 - 0.55 * d ** 2, 0, 1)[..., None]
        self.grain = rng.normal(0, 1, (8, H // 2, W // 2)).astype(np.float32)

    # world positions of all string nodes
    def string_points(self):
        cx, cy = CENTER
        n = np.linspace(0, 1, STRING_NODES)
        rad = STRING_INNER + (STRING_OUTER - STRING_INNER) * n
        pts = []
        for k in range(12):
            ang = k * (2 * np.pi / 12) - np.pi / 2
            # gentle spiral so strings read as a vortex, not spokes
            ang = ang + 0.55 * n
            ux, uy = np.cos(ang), np.sin(ang)
            nx, ny = -uy, ux
            x = cx + ux * rad + nx * self.u[k]
            y = cy + uy * rad + ny * self.u[k]
            pts.append((x, y))
        return pts

    def step(self, i):
        F = self.F
        t = i * DT
        cx, cy = CENTER
        bass = 0.6 * F["sub"][i] + 0.4 * F["bass"][i]
        sec = int(F["section"][i])
        sec_e = self.sec_energy[sec]
        calm = np.clip(1.0 - sec_e / 0.6, 0, 1)          # 1 in breakdowns
        gm = (G_BASE + G_BASS * bass) * (1.0 - 0.45 * calm)

        kick_imp = 0.0
        if i in self.kick_set:
            kick_imp = KICK_PUSH * (0.4 + 0.6 * self.kick_str[i])

        # snare -> new shockwave, alternating either side of the core
        if i in self.snare_set and F["snare"][i] > 0.25:
            j = self.nrings if self.nrings < 16 else int(np.argmax(self.rings[:16, 4]))
            side = 1 if self.ring_side % 2 == 0 else -1
            self.ring_side += 1
            off = 160 + 140 * self.rng.random()
            ang = self.theta + side * 1.2 + self.rng.normal(0, 0.3)
            self.rings[j] = (cx + np.cos(ang) * off, cy + np.sin(ang) * off, 0.0,
                             SNARE_PUSH * (0.5 + 0.5 * self.snare_set[i]), 0.0)
            self.nrings = min(16, self.nrings + 1)
        # advance rings, drop old ones
        self.rings[:self.nrings, 2] += SNARE_SPEED * DT
        self.rings[:self.nrings, 4] += DT
        keep = self.rings[:self.nrings, 4] < 1.6
        r = self.rings[:self.nrings][keep]
        self.nrings = len(r)
        self.rings[:self.nrings] = r

        torque = 0.85 + TORQUE * F["lowmid"][i]
        turb = TURB * (0.15 + 0.6 * F["air"][i] + 0.9 * F["hat"][i])
        drag = DRAG * (1.0 - 0.5 * calm)
        rnd = self.rng.random((N_PARTICLES, 2))
        step_particles(self.px, self.py, self.vx, self.vy, cx, cy, gm, kick_imp,
                       torque, turb, drag, self.rings, self.nrings, t, DT, rnd,
                       self.home, 1.0 + 0.6 * calm)

        # strings: drive by chroma, pluck on rising notes
        ch = F["chroma"][i]
        harm = F["harmonic"][i] * (0.4 + 0.6 * F["tonal"][i])
        amp = (ch ** 2.0) * harm * 9000.0
        rise = np.clip(ch - self.prev_chroma, 0, None)
        self.uv[:, 10:20] += (rise[:, None] * 900.0 * harm) * np.hanning(10)[None, :]
        self.prev_chroma = ch * 0.7 + self.prev_chroma * 0.3
        step_strings(self.u, self.uv, amp, self.harm, t, DT, 8)
        np.clip(self.u, -140, 140, out=self.u)

        # sparks on hats, born on vibrating strings
        if i in self.hat_set:
            nsp = int(40 + 160 * self.hat_set[i] * (0.3 + F["air"][i]))
            pts = self.string_points()
            for _ in range(nsp):
                k = self.rng.integers(12)
                n = self.rng.integers(8, STRING_NODES - 2)
                p = self.spark_ptr
                self.spark_ptr = (p + 1) % N_SPARKS
                self.sx[p], self.sy[p] = pts[k][0][n], pts[k][1][n]
                a = self.rng.uniform(0, 2 * np.pi)
                sp = self.rng.uniform(150, 650)
                self.svx[p], self.svy[p] = np.cos(a) * sp, np.sin(a) * sp
                life = self.rng.uniform(0.35, 0.9)
                self.slife[p] = life; self.smax[p] = life
                self.scol[p] = 0.5 * STRING_COLORS[k] + 0.5
        step_sparks(self.sx, self.sy, self.svx, self.svy, self.slife, cx, cy, gm,
                    turb, t, DT)

        # camera + palette
        self.theta += (SPIN_BASE + SPIN_ENERGY * F["energy_slow"][i]) * DT
        target = np.array(PALETTES[sec % len(PALETTES)], float)
        self.pal += (target - self.pal) * (1 - np.exp(-1.2 * DT))
        self.calm = calm

    def draw(self, i):
        F = self.F
        cx, cy = CENTER
        buf = self.buf
        buf *= TRAIL
        bg, cold, hot, accent = self.pal
        ca, sa = math.cos(self.theta), math.sin(self.theta)
        zoom = 1.0 + 0.035 * F["kick"][i] + 0.04 * F["energy_slow"][i]

        particle_colors(self.vx, self.vy, cold.astype(np.float32), hot.astype(np.float32),
                        float(F["centroid"][i]), self.pcol, self.pint,
                        PARTICLE_INT * (0.6 + 0.6 * F["rms"][i]))
        splat(buf, self.px, self.py, self.pcol, self.pint, cx, cy, ca, sa, zoom)

        # shockwave rings
        for j in range(self.nrings):
            rcx, rcy, rad, push, age = self.rings[j]
            if rad < 4:
                continue
            m = int(2 * np.pi * rad / 1.2)
            a = np.linspace(0, 2 * np.pi, m, endpoint=False)
            # rings wobble slightly: a refracted, imperfect wavefront
            rr = rad * (1 + 0.012 * np.sin(a * 7 + age * 9))
            xs = rcx + np.cos(a) * rr
            ys = rcy + np.sin(a) * rr
            fade = (1 - age / 1.6) ** 2 * (push / SNARE_PUSH)
            cols = np.tile(accent.astype(np.float32), (m, 1))
            ints = np.full(m, 0.16 * fade, np.float32)
            splat(buf, xs, ys, cols, ints, cx, cy, ca, sa, zoom)

        # strings
        pts = self.string_points()
        ch = F["chroma"][i]
        for k in range(12):
            x, y = pts[k]
            seg = np.hypot(np.diff(x), np.diff(y))
            L = np.concatenate([[0], np.cumsum(seg)])
            s = np.arange(0, L[-1], 1.6)
            xs = np.interp(s, L, x); ys = np.interp(s, L, y)
            taper = np.sin(np.pi * s / L[-1]) ** 0.6
            disp = np.interp(s, L, np.abs(self.u[k]))
            ints = (STRING_INT * (0.06 + 0.5 * ch[k] ** 2) * taper *
                    (0.5 + disp / 40.0)).astype(np.float32)
            cols = np.tile(STRING_COLORS[k], (len(s), 1))
            splat(buf, xs, ys, cols, ints, cx, cy, ca, sa, zoom)

        # sparks (drawn as short streaks along velocity)
        alive = self.slife > 0
        if alive.any():
            idx = np.nonzero(alive)[0]
            fr = (self.slife[idx] / self.smax[idx]).astype(np.float32)
            for back in np.linspace(0, 0.03, 10):
                xs = self.sx[idx] - self.svx[idx] * back
                ys = self.sy[idx] - self.svy[idx] * back
                splat(buf, xs, ys, self.scol[idx],
                      (0.45 * fr * (1 - back / 0.035)).astype(np.float32),
                      cx, cy, ca, sa, zoom)

        # core: hot singularity glow that flares on kicks
        # core: a dark well ringed by a bright 'photon ring' that swells with
        # the sub and flares on kicks
        pr = 46.0 * (1 + 0.25 * F["sub"][i] + 0.18 * F["kick"][i])
        m = 900
        a = np.linspace(0, 2 * np.pi, m, endpoint=False)
        doppler = 0.55 + 0.45 * np.cos(a - 0.6)          # brighter approaching side
        flare = 0.10 + 0.25 * F["kick"][i] + 0.1 * F["sub"][i]
        for dr, w in ((0, 1.0), (2.5, 0.5), (-2.5, 0.5), (6, 0.2)):
            xs = cx + np.cos(a) * (pr + dr)
            ys = cy + np.sin(a) * (pr + dr) * 0.92
            cols = np.tile((hot * 0.6 + 0.4).astype(np.float32), (m, 1))
            splat(buf, xs, ys, cols, (flare * w * doppler).astype(np.float32),
                  cx, cy, 1.0, 0.0, zoom)
        cv2.circle(buf, (int(cx), int(cy)), int(pr * zoom * 0.8), (0, 0, 0), -1,
                   lineType=cv2.LINE_8)

        return self.compose(i)

    def compose(self, i):
        F = self.F
        bg = self.pal[0]
        img = self.buf
        # bloom at two scales
        small = cv2.resize(img, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        b1 = cv2.GaussianBlur(small, (0, 0), 4)
        tiny = cv2.resize(small, (W // 16, H // 16), interpolation=cv2.INTER_AREA)
        b2 = cv2.GaussianBlur(tiny, (0, 0), 3)
        bloom = (cv2.resize(b1, (W, H), interpolation=cv2.INTER_LINEAR) * 0.9 +
                 cv2.resize(b2, (W, H), interpolation=cv2.INTER_LINEAR) * 1.6) * GLOW
        lift = 0.5 + 1.2 * F["energy_slow"][i] + 0.6 * F["bass"][i]
        x = img + bloom + self.bg_shape * (np.asarray(bg, np.float32) * lift * 0.7)
        x = 1.0 - np.exp(-x * EXPOSURE)
        x *= self.vignette
        t = i * DT
        dur = F["duration"]
        fade = min(1.0, t / FADE_IN, max(0.0, (dur - t) / FADE_OUT))
        x *= fade
        x = np.clip(x, 0, 1) ** (1 / 1.5)      # approx linear -> display gamma
        g = cv2.resize(self.grain[i % 8], (W, H), interpolation=cv2.INTER_NEAREST)
        x += g[..., None] * (1.2 / 255)
        return (np.clip(x, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--out", default="out.mp4")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--stills", default=None, help="comma list of seconds -> PNGs")
    ap.add_argument("--crf", type=int, default=19,
                    help="H.264 quality: 19 = master (~3.7 MB/s), 25 = share (~1 MB/s)")
    a = ap.parse_args()

    F = dict(np.load(a.features))
    global FPS, DT
    FPS = int(F["fps"])
    DT = 1.0 / FPS
    n = int(F["nframes"])
    sc = Scene(F)

    if a.stills:
        want = sorted(int(float(s) * FPS) for s in a.stills.split(","))
        for i in range(max(want) + 1):
            sc.step(i)
            if any(0 <= w - i < 12 for w in want):   # draw ~0.4 s of trails before
                frame = sc.draw(i)
                if i in want:
                    p = f"still_{i / FPS:06.2f}.png"
                    cv2.imwrite(p, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    print("wrote", p, flush=True)
        return

    f0 = int(a.start * FPS)
    f1 = n if a.end is None else min(n, int(a.end * FPS))
    tmp = a.out[:-4] + ".partial.mp4"
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
         "-preset", "veryfast", "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-g", "60", tmp],
        stdin=subprocess.PIPE)
    warm = max(0, f0 - 15)
    for i in range(f1):
        sc.step(i)
        if i >= warm:
            frame = sc.draw(i)
            if i >= f0:
                ff.stdin.write(frame.tobytes())
        if i % 150 == 0:
            print(f"frame {i}/{f1}", flush=True)
    ff.stdin.close()
    if ff.wait() == 0:
        os.replace(tmp, a.out)
        print("done", a.out)
    else:
        sys.exit("ffmpeg failed")


if __name__ == "__main__":
    main()
