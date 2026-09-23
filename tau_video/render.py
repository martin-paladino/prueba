"""Renderiza el video de la proteína tau cuadro por cuadro (Pillow + ffmpeg).

Estética inspirada en ilustración plana "dibujada a mano": fondo de papel,
contornos marrón oscuro, verde menta / naranja / rosa, zooms lentos y rayos de luz.

Uso: python3 render.py --build build/ --out salida.mp4 [--workers 4] [--only escena] [--frame t]
"""
import argparse
import json
import math
import os
import subprocess
import sys
from multiprocessing import Pool

import numpy as np
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from script import SCENES  # noqa: E402

W, H = 1280, 720
SS = 2                      # supersampling para bordes suaves
FPS = 30
XFADE = 0.9                 # duración del fundido entre escenas
ROOT = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(ROOT, "..", "assets", "fonts")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# ----------------------------------------------------------------------------
# Paleta
# ----------------------------------------------------------------------------
INK = (59, 42, 32)
PAPER = (239, 232, 216)
PAPER_D = (226, 216, 196)
MINT = (169, 207, 192)
MINT_D = (94, 158, 143)
MINT_L = (205, 229, 218)
ORANGE = (231, 161, 59)
ORANGE_D = (196, 116, 40)
ORANGE_L = (246, 200, 118)
PINK = (217, 115, 107)
PINK_D = (170, 74, 70)
PINK_L = (236, 180, 168)
BLUE = (143, 193, 201)
CYTO = (178, 212, 199)
CYTO_D = (150, 190, 176)
DARK = (58, 84, 88)
DARK_D = (34, 52, 58)
SPACE = (17, 31, 52)
TUBE = (200, 226, 216)
TUB_A = (138, 192, 184)
TUB_B = (84, 150, 142)
TAU = (236, 158, 52)
TAU_BAD = (206, 88, 52)
P_YEL = (248, 212, 80)
KIN = (116, 146, 204)
KINASE = (164, 128, 196)
PHOSPHATASE = (124, 182, 118)
PLAQUE = (160, 118, 78)
CREAM = (251, 246, 232)
GRAY = (150, 150, 146)


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def smooth(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


def ramp(t, t0, t1):
    """0 antes de t0, 1 después de t1, suave entre medio."""
    if t1 <= t0:
        return 1.0 if t >= t0 else 0.0
    return smooth((t - t0) / (t1 - t0))


def lerp(a, b, k):
    return a + (b - a) * k


def mix(c1, c2, k):
    return tuple(int(round(lerp(a, b, k))) for a, b in zip(c1, c2))


def rgba(c, a=1.0):
    return (c[0], c[1], c[2], int(clamp(a) * 255))


# ----------------------------------------------------------------------------
# Fuentes
# ----------------------------------------------------------------------------
_FONTS = {}


def font(name, px, weight="SemiBold"):
    key = (name, int(px), weight)
    if key not in _FONTS:
        f = ImageFont.truetype(os.path.join(FONT_DIR, name + ".ttf"), max(4, int(px)))
        try:
            f.set_variation_by_name(weight)
        except Exception:
            pass
        _FONTS[key] = f
    return _FONTS[key]


# ----------------------------------------------------------------------------
# Lienzo con cámara
# ----------------------------------------------------------------------------
class Canvas:
    def __init__(self, img, cx=0.0, cy=0.0, z=1.0):
        self.img = img
        self.d = ImageDraw.Draw(img, "RGBA")
        self.cx, self.cy, self.z = cx, cy, z

    def P(self, x, y):
        return ((x - self.cx) * self.z + W / 2) * SS, ((y - self.cy) * self.z + H / 2) * SS

    def s(self, v):
        return v * self.z * SS

    def wx(self, sx):
        return self.cx + (sx - W / 2) / self.z

    def wy(self, sy):
        return self.cy + (sy - H / 2) / self.z

    def visible(self, x, y, r):
        px, py = self.P(x, y)
        rr = self.s(r)
        return -rr < px < W * SS + rr and -rr < py < H * SS + rr

    def lw(self, w):
        return max(1, int(round(self.s(w))))

    def circle(self, x, y, r, fill=None, outline=INK, w=2.5, a=1.0):
        px, py = self.P(x, y)
        rr = self.s(r)
        if rr < 0.5:
            return
        self.d.ellipse([px - rr, py - rr, px + rr, py + rr],
                       fill=rgba(fill, a) if fill else None,
                       outline=rgba(outline, a) if outline else None,
                       width=self.lw(w) if outline else 0)

    def ellipse(self, x, y, rx, ry, fill=None, outline=INK, w=2.5, a=1.0):
        px, py = self.P(x, y)
        self.d.ellipse([px - self.s(rx), py - self.s(ry), px + self.s(rx), py + self.s(ry)],
                       fill=rgba(fill, a) if fill else None,
                       outline=rgba(outline, a) if outline else None,
                       width=self.lw(w) if outline else 0)

    def poly(self, pts, fill=None, outline=INK, w=2.5, a=1.0):
        pp = [self.P(x, y) for x, y in pts]
        if fill:
            self.d.polygon(pp, fill=rgba(fill, a))
        if outline:
            self.d.line(pp + [pp[0]], fill=rgba(outline, a), width=self.lw(w), joint="curve")

    def line(self, pts, color=INK, w=2.5, a=1.0, caps=True):
        pp = [self.P(x, y) for x, y in pts]
        lw = self.lw(w)
        self.d.line(pp, fill=rgba(color, a), width=lw, joint="curve")
        if caps and lw > 3:
            r = lw / 2
            for px, py in (pp[0], pp[-1]):
                self.d.ellipse([px - r, py - r, px + r, py + r], fill=rgba(color, a))

    def stroke(self, pts, color, w, ow=2.2, a=1.0, ink=INK):
        """Trazo con contorno (como un tubo dibujado)."""
        self.line(pts, ink, w + 2 * ow, a)
        self.line(pts, color, w, a)

    def rrect(self, x0, y0, x1, y1, r, fill=None, outline=INK, w=2.5, a=1.0):
        p0 = self.P(x0, y0)
        p1 = self.P(x1, y1)
        self.d.rounded_rectangle([p0[0], p0[1], p1[0], p1[1]], radius=self.s(r),
                                 fill=rgba(fill, a) if fill else None,
                                 outline=rgba(outline, a) if outline else None,
                                 width=self.lw(w) if outline else 0)

    def text(self, x, y, txt, size, color=INK, a=1.0, fnt="Fredoka", weight="SemiBold", anchor="mm"):
        px, py = self.P(x, y)
        self.d.text((px, py), txt, font=font(fnt, self.s(size), weight), fill=rgba(color, a), anchor=anchor)

    def glow(self, x, y, r, color, a=0.5, steps=10):
        for i in range(steps):
            k = 1 - i / steps
            self.circle(x, y, r * k, fill=color, outline=None, a=a * (1 - k) ** 0.8 / steps * 2.2)


# ----------------------------------------------------------------------------
# Geometría auxiliar
# ----------------------------------------------------------------------------
def catmull(pts, n=8, closed=True):
    out = []
    m = len(pts)
    rng = range(m) if closed else range(m - 1)
    for i in rng:
        p0 = pts[(i - 1) % m] if closed or i > 0 else pts[0]
        p1 = pts[i]
        p2 = pts[(i + 1) % m]
        p3 = pts[(i + 2) % m] if closed or i + 2 < m else pts[-1]
        for k in range(n):
            t = k / n
            t2, t3 = t * t, t * t * t
            out.append(tuple(0.5 * ((2 * p1[j]) + (-p0[j] + p2[j]) * t
                                    + (2 * p0[j] - 5 * p1[j] + 4 * p2[j] - p3[j]) * t2
                                    + (-p0[j] + 3 * p1[j] - 3 * p2[j] + p3[j]) * t3) for j in range(2)))
    if not closed:
        out.append(pts[-1])
    return out


def blob(x, y, r, seed=0, t=0.0, wob=0.05, n=56, sx=1.0, sy=1.0):
    rng = np.random.default_rng(seed)
    ks = [2, 3, 4, 5]
    amps = rng.uniform(0.3, 1.0, len(ks)) * wob
    phs = rng.uniform(0, 2 * np.pi, len(ks))
    pts = []
    for i in range(n):
        a = 2 * math.pi * i / n
        rr = r * (1 + sum(am * math.sin(k * a + ph + t * 0.6 * (j + 1))
                          for j, (k, am, ph) in enumerate(zip(ks, amps, phs))))
        pts.append((x + math.cos(a) * rr * sx, y + math.sin(a) * rr * sy))
    return pts


def inside(pt, poly):
    x, y = pt
    c = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            c = not c
        j = i
    return c


def path_point(path, s):
    """Punto en una polilínea a fracción s de su longitud."""
    segs = [math.dist(path[i], path[i + 1]) for i in range(len(path) - 1)]
    L = sum(segs)
    d = clamp(s) * L
    for i, sl in enumerate(segs):
        if d <= sl or i == len(segs) - 1:
            k = d / sl if sl else 0
            return (lerp(path[i][0], path[i + 1][0], k), lerp(path[i][1], path[i + 1][1], k))
        d -= sl
    return path[-1]


# ----------------------------------------------------------------------------
# Texturas globales (grano de papel + viñeta), fondos precalculados
# ----------------------------------------------------------------------------
_POST = None
_RADIAL = None


def post_map():
    global _POST
    if _POST is None:
        rng = np.random.default_rng(11)
        grain = rng.normal(0, 1, (H, W)).astype(np.float32)
        # Grano con algo de "fibra": suavizado horizontal leve.
        g2 = (grain + np.roll(grain, 1, 1) + np.roll(grain, -1, 1)) / 3
        blotch = rng.normal(0, 1, (H // 40 + 2, W // 40 + 2)).astype(np.float32)
        blotch = np.array(Image.fromarray(blotch).resize((W, H), Image.BICUBIC))
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
        vig = 1 - 0.22 * np.clip(r - 0.45, 0, None) ** 1.6
        _POST = (vig * (1 + 0.035 * g2 + 0.02 * blotch))[..., None]
    return _POST


def radial_mask():
    global _RADIAL
    if _RADIAL is None:
        w, h = W // 4, H // 4
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx - w / 2) / (w * 0.55)) ** 2 + ((yy - h / 2) / (h * 0.75)) ** 2)
        m = np.clip(1 - r, 0, 1) ** 1.3
        _RADIAL = Image.fromarray((m * 255).astype(np.uint8)).resize((W * SS, H * SS), Image.BICUBIC)
    return _RADIAL


def bg_gradient(img, c_center, c_edge):
    img.paste(Image.new("RGB", img.size, c_edge))
    img.paste(Image.new("RGB", img.size, c_center), (0, 0), radial_mask())


def rays(cv, t, color, a=0.25, n=48, rot=0.02, x=None, y=None, r0=60, length=1600, w=1.4):
    """Rayos de luz finos desde un punto (en coordenadas de pantalla si x=None)."""
    if x is None:
        px, py = W * SS / 2, H * SS / 2
    else:
        px, py = cv.P(x, y)
    rng = np.random.default_rng(5)
    offs = rng.uniform(0, 1, n)
    lens = rng.uniform(0.5, 1.0, n)
    for i in range(n):
        ang = 2 * math.pi * (i + offs[i] * 0.6) / n + t * rot
        rr0 = r0 * SS
        rr1 = length * SS * lens[i]
        cvx, cvy = math.cos(ang), math.sin(ang)
        cv.d.line([(px + cvx * rr0, py + cvy * rr0), (px + cvx * rr1, py + cvy * rr1)],
                  fill=rgba(color, a * (0.5 + 0.5 * math.sin(t * 0.7 + i * 1.3) ** 2)),
                  width=max(1, int(w * SS)))


def rings(cv, t, color, a=0.25, x=None, y=None):
    if x is None:
        px, py = W * SS / 2, H * SS / 2
    else:
        px, py = cv.P(x, y)
    for i in range(7):
        r = (120 + i * 90 + (t * 12) % 90) * SS
        cv.d.ellipse([px - r, py - r, px + r, py + r], outline=rgba(color, a * (1 - i / 7)), width=SS)


def particles(cv, t, seed, n, area, cols, rmin=4, rmax=14, par=0.6, drift=6, dark=0.0):
    """Partículas flotantes del citoplasma (con leve paralaje)."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(-area, area, n)
    ys = rng.uniform(-area * 0.6, area * 0.6, n)
    rs = rng.uniform(rmin, rmax, n)
    ph = rng.uniform(0, 6.28, n)
    kinds = rng.integers(0, len(cols), n)
    for i in range(n):
        x = xs[i] + math.sin(t * 0.3 + ph[i]) * drift
        y = ys[i] + math.cos(t * 0.25 + ph[i]) * drift
        # Paralaje: la partícula se mueve menos que la cámara.
        px = cv.cx + (x - cv.cx * par)
        py = cv.cy + (y - cv.cy * par)
        col = cols[kinds[i]]
        if col is None:
            cv.circle(px, py, rs[i] * 0.45, fill=mix(INK, DARK_D, dark), outline=None, a=0.85)
        else:
            cv.circle(px, py, rs[i], fill=col, outline=mix(INK, DARK_D, dark), w=1.6, a=0.9)


def label(cv, tx, ty, txt, t, t_in, dx=0, dy=-80, size=26, t_out=None, color=CREAM, tcolor=INK, dot=True):
    """Etiqueta con línea guía. (tx,ty) en coordenadas del mundo, desplazamiento en píxeles."""
    a = ramp(t, t_in, t_in + 0.5)
    if t_out is not None:
        a *= 1 - ramp(t, t_out, t_out + 0.5)
    if a <= 0.01:
        return
    px, py = cv.P(tx, ty)
    pop = 0.85 + 0.15 * smooth((t - t_in) / 0.35)
    lx, ly = px + dx * SS, py + dy * SS
    f = font("Fredoka", size * SS * pop, "SemiBold")
    bb = cv.d.textbbox((0, 0), txt, font=f, anchor="lt")
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    padx, pady = 16 * SS, 10 * SS
    if dx != 0 or dy != 0:
        cv.d.line([(px, py), (lx, ly)], fill=rgba(INK, a), width=int(2.2 * SS))
        if dot:
            r = 5 * SS
            cv.d.ellipse([px - r, py - r, px + r, py + r], fill=rgba(INK, a))
    box = [lx - tw / 2 - padx, ly - th / 2 - pady, lx + tw / 2 + padx, ly + th / 2 + pady]
    cv.d.rounded_rectangle([box[0] + 3 * SS, box[1] + 4 * SS, box[2] + 3 * SS, box[3] + 4 * SS],
                           radius=14 * SS, fill=rgba(INK, a * 0.35))
    cv.d.rounded_rectangle(box, radius=14 * SS, fill=rgba(color, a), outline=rgba(INK, a), width=int(2.4 * SS))
    cv.d.text((lx, ly), txt, font=f, fill=rgba(tcolor, a), anchor="mm")


def title(cv, txt, y, size, a, color=INK, sub=None, subsize=None, shadow=True):
    if a <= 0.01:
        return
    f = font("Fredoka", size * SS, "SemiBold")
    px, py = W * SS / 2, y * SS
    if shadow:
        cv.d.text((px + 3 * SS, py + 4 * SS), txt, font=f, fill=rgba(INK, a * 0.25), anchor="mm")
    cv.d.text((px, py), txt, font=f, fill=rgba(color, a), anchor="mm")
    if sub:
        f2 = font("Nunito", (subsize or size * 0.4) * SS, "Bold")
        cv.d.text((px, py + size * 0.85 * SS), sub, font=f2, fill=rgba(color, a * 0.9), anchor="mm")


# ----------------------------------------------------------------------------
# Elementos biológicos
# ----------------------------------------------------------------------------
def nucleus(cv, x, y, r, seed=1, t=0.0, a=1.0, col=ORANGE, cold=ORANGE_D, coll=ORANGE_L):
    cv.poly(blob(x, y, r, seed, t, 0.02), fill=col, outline=INK, w=2.6, a=a)
    rng = np.random.default_rng(seed + 10)
    # Cromatina: trazos cortos, como en la referencia.
    for _ in range(int(r * 1.1)):
        ang = rng.uniform(0, 6.28)
        rr = r * math.sqrt(rng.uniform(0, 0.8))
        px, py = x + math.cos(ang) * rr, y + math.sin(ang) * rr
        l = r * rng.uniform(0.06, 0.13)
        b = rng.uniform(0, 6.28)
        cv.line([(px, py), (px + math.cos(b) * l, py + math.sin(b) * l)], cold, w=r * 0.03 + 0.8, a=a * 0.9)
    cv.circle(x + r * 0.28, y + r * 0.2, r * 0.2, fill=cold, outline=None, a=a * 0.8)
    cv.ellipse(x - r * 0.4, y - r * 0.45, r * 0.22, r * 0.14, fill=coll, outline=None, a=a * 0.7)


def mitochondrion(cv, x, y, length, width, ang=0.0, a=1.0, col=PINK, cold=PINK_D):
    ca, sa = math.cos(ang), math.sin(ang)

    def T(u, v):
        return (x + u * ca - v * sa, y + u * sa + v * ca)
    pts = []
    for i in range(24):
        th = math.pi / 2 + math.pi * i / 23
        pts.append(T(-length / 2 + width / 2 + math.cos(th) * width / 2, math.sin(th) * width / 2))
    for i in range(24):
        th = -math.pi / 2 + math.pi * i / 23
        pts.append(T(length / 2 - width / 2 + math.cos(th) * width / 2, math.sin(th) * width / 2))
    cv.poly(pts, fill=col, outline=INK, w=2.4, a=a)
    zz = []
    n = 7
    for i in range(n + 1):
        u = -length / 2 + width * 0.55 + (length - width * 1.1) * i / n
        zz.append(T(u, (width * 0.28) * (1 if i % 2 else -1)))
    cv.line(zz, cold, w=width * 0.1 + 1, a=a)


def vesicle(cv, x, y, r, col=BLUE, a=1.0):
    cv.circle(x, y, r, fill=col, outline=INK, w=2.2, a=a)
    cv.circle(x - r * 0.35, y - r * 0.35, r * 0.25, fill=(255, 255, 255), outline=None, a=a * 0.45)


class Microtubule:
    """Microtúbulo horizontal: columnas de dímeros de tubulina (α claro / β oscuro).

    Para desarmarlo se parte en tramos que se despolimerizan desde los extremos,
    con los protofilamentos "pelándose" hacia afuera, como ocurre en la realidad.
    """
    STEP = 12.0
    R = 30.0
    ROWS = (-18.0, -6.0, 6.0, 18.0)

    def __init__(self, x0, x1, y, seed=0):
        self.x0, self.x1, self.y = x0, x1, y
        self.n = int((x1 - x0) / self.STEP)
        rng = np.random.default_rng(seed)
        bounds = [0]
        while bounds[-1] < self.n:
            bounds.append(bounds[-1] + int(rng.integers(22, 40)))
        bounds[-1] = self.n
        self.seg_of = np.zeros(self.n, int)
        self.e = np.zeros(self.n)
        self.side = np.zeros(self.n)
        for k in range(len(bounds) - 1):
            a, b = bounds[k], bounds[k + 1]
            for ci in range(a, b):
                self.seg_of[ci] = k
                self.e[ci] = min(ci - a, b - 1 - ci) / max(1, b - a)
                self.side[ci] = -1 if (ci - a) < (b - a) / 2 else 1
        ns = len(bounds)
        self.seg_dx = rng.normal(0, 1, ns)
        self.seg_dy = rng.normal(0, 1, ns)
        self.vy = rng.normal(0, 1, (self.n, len(self.ROWS)))

    def col_x(self, c):
        return self.x0 + c * self.STEP

    def draw(self, cv, t, destroy=0.0, a=1.0, dark=0.0):
        R, st = self.R, self.STEP
        ink = mix(INK, DARK_D, dark * 0.5)
        tube = mix(TUBE, (130, 160, 158), dark * 0.6)
        ta, tb = mix(TUB_A, (104, 146, 140), dark * 0.5), mix(TUB_B, (72, 116, 110), dark * 0.5)
        left = cv.cx - W / 2 / cv.z - 60
        right = cv.cx + W / 2 / cv.z + 60
        c0 = max(0, int((left - self.x0) / st))
        c1 = min(self.n, int((right - self.x0) / st) + 1)
        lim = destroy * 0.44
        age = np.clip((lim - self.e) * 7, 0, 1)
        broken = age > 0
        segoff = {}
        for k in set(self.seg_of[c0:c1].tolist()):
            segoff[k] = (self.seg_dx[k] * destroy * 20, self.seg_dy[k] * destroy * 28)
        # Cuerpo: tramos contiguos intactos (dentro del mismo segmento).
        runs = []
        cur = None
        for ci in range(c0, c1):
            ok = not broken[ci]
            if ok and cur is not None and destroy > 0.01 and self.seg_of[ci] != self.seg_of[cur[0]]:
                runs.append(cur)
                cur = None
            if ok:
                cur = [ci, ci] if cur is None else [cur[0], ci]
            elif cur is not None:
                runs.append(cur)
                cur = None
        if cur is not None:
            runs.append(cur)
        for r0, r1 in runs:
            ox, oy = segoff[self.seg_of[r0]]
            xa, xb = self.col_x(r0) - st / 2 + ox, self.col_x(r1) + st / 2 + ox
            cv.rrect(xa, self.y - R + oy, xb, self.y + R + oy, 10, fill=tube, outline=ink, w=2.6, a=a)
        for ci in range(c0, c1):
            ox, oy = segoff[self.seg_of[ci]]
            x = self.col_x(ci) + ox
            for ri, ry in enumerate(self.ROWS):
                col = ta if ci % 2 == 0 else tb
                col = mix(col, (255, 255, 255), 0.22) if ri == 0 else mix(col, ink, 0.22) if ri == 3 else col
                bx, by, al = x + ri * 2.5, self.y + ry + oy, a
                if broken[ci]:
                    ag = age[ci]
                    # Protofilamentos que se curvan hacia afuera y se sueltan.
                    bx += self.side[ci] * ag * 26
                    by += (ry / 18.0) * ag ** 1.4 * 55 + self.vy[ci, ri] * ag * 25
                    al = a * (1 - 0.8 * ag)
                cv.rrect(bx - 5.0, by - 4.8, bx + 5.0, by + 4.8, 3.2, fill=col, outline=ink, w=0.9, a=al)
        # Brillo superior / sombra inferior (volumen de tubo)
        for r0, r1 in runs:
            ox, oy = segoff[self.seg_of[r0]]
            xa, xb = self.col_x(r0) + ox, self.col_x(r1) + ox
            if xb - xa > 20:
                cv.line([(xa + 4, self.y - R + 5 + oy), (xb - 4, self.y - R + 5 + oy)], (255, 255, 255), w=3, a=a * 0.35)
                cv.line([(xa + 4, self.y + R - 4 + oy), (xb - 4, self.y + R - 4 + oy)], ink, w=3, a=a * 0.18)
        return broken


def tau_points(cx, cy, k_bound, t, seed=0, side=-1, length=86, tail=48, wig=1.0):
    """Puntos de una molécula de tau. k_bound=1 unida (a lo largo de la superficie), 0 libre."""
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 6.28)
    rot = rng.uniform(-0.6, 0.6)
    n = 16
    pts = []
    for i in range(n):
        u = i / (n - 1)
        # Unida: dominio de unión a lo largo del microtúbulo + cola (dominio de proyección).
        if u < 0.65:
            uu = u / 0.65
            bx = cx - length / 2 + length * uu
            by = cy + math.sin(uu * 9 + ph) * 3.2
        else:
            uu = (u - 0.65) / 0.35
            bx = cx + length / 2 + math.sin(uu * 2.2) * 10
            by = cy + side * tail * uu + math.sin(t * 2 + uu * 5 + ph) * 6 * uu
        # Libre: desordenada (intrínsecamente desordenada), ondulante.
        ang = u * 5.5 + ph + math.sin(t * 1.3 + ph) * 0.8 * wig
        fr = 38 + 10 * math.sin(u * 3 + t * 0.9 + ph)
        fx = cx + math.cos(ang + rot) * fr * (0.3 + u) * 0.8 + (u - 0.5) * 30
        fy = cy + math.sin(ang * 1.3 + rot) * fr * (0.3 + u) * 0.6
        pts.append((lerp(fx, bx, k_bound), lerp(fy, by, k_bound)))
    return pts


def draw_tau(cv, pts, col=TAU, a=1.0, w=6.5, phos=0, pk=1.0, dark=0.0):
    ink = mix(INK, DARK_D, dark)
    cv.stroke(pts, col, w, ow=2.3, a=a, ink=ink)
    # Fosfatos sobre la cola / región rica en prolina.
    idx = [11, 13, 15, 8, 6, 3]
    for j in range(int(math.ceil(phos))):
        kk = clamp(phos - j) if j == int(math.ceil(phos)) - 1 else 1.0
        kk = kk * pk
        if kk <= 0.02:
            continue
        x, y = pts[idx[j % len(idx)]]
        off = 13 * (1 if j % 2 else -1)
        px, py = x + off * 0.4, y + off
        cv.line([(x, y), (px, py)], ink, w=2, a=a * kk)
        r = 10.5 * (0.6 + 0.4 * kk)
        cv.circle(px, py, r, fill=P_YEL, outline=ink, w=2, a=a * kk)
        cv.text(px, py + 0.5, "P", 13 * (0.6 + 0.4 * kk), ink, a=a * kk, weight="Bold")


def motor(cv, x, ytop, t, cargo="vesicle", color=KIN, a=1.0, speed_phase=0.0, stall=0.0, flip=False):
    """Proteína motora caminando sobre el microtúbulo (cabezas alternadas)."""
    period = 0.7
    ph = (t / period + speed_phase)
    step = 18
    k = ph % 1.0
    lift = math.sin(k * math.pi) * (1 - stall)
    fa = x - step / 2 + (step if int(ph) % 2 else 0) * 0
    # Pie trasero se mueve adelante en cada medio ciclo
    front = x + step / 2
    back = x - step / 2
    moving = back + (front + step - back) * smooth(k)
    fx1, fx2 = (moving, front) if int(ph) % 2 == 0 else (front, moving)
    sgn = -1 if flip else 1
    hip = (x, ytop - 30 * sgn)
    for fx, lft in ((fx1, lift if int(ph) % 2 == 0 else 0), (fx2, lift if int(ph) % 2 else 0)):
        fy = ytop - 7 * sgn - lft * 12 * sgn
        cv.stroke([hip, ((hip[0] + fx) / 2, (hip[1] + fy) / 2 - 3 * sgn), (fx, fy)], color, 4.5, ow=1.8, a=a)
        cv.ellipse(fx, fy, 9, 6.5, fill=color, outline=INK, w=2, a=a)
    top = (x, ytop - 70 * sgn)
    cv.stroke([hip, (x + 3, (hip[1] + top[1]) / 2), top], color, 4, ow=1.8, a=a)
    if cargo == "vesicle":
        vesicle(cv, x, ytop - 100 * sgn, 34, col=PINK_L, a=a)
    elif cargo == "mito":
        mitochondrion(cv, x, ytop - 95 * sgn, 120, 50, 0, a=a)
    fa = fa


# ------------------------------- Neurona ------------------------------------
class Neuron:
    def __init__(self, x, y, scale=1.0, seed=0, axon_len=900, axon_dir=0.0, soma_r=80, n_dend=6,
                 axon_curve=40, terminal=True):
        self.x, self.y, self.sc = x, y, scale
        self.soma_r = soma_r
        rng = np.random.default_rng(seed)
        self.seed = seed
        self.branches = []  # (points, width)

        def grow(px, py, ang, length, width, depth):
            pts = [(px, py)]
            n = 6
            a = ang
            for i in range(n):
                a += rng.normal(0, 0.18)
                px += math.cos(a) * length / n
                py += math.sin(a) * length / n
                pts.append((px, py))
            self.branches.append((pts, width))
            if depth > 0:
                for s in (-1, 1):
                    if rng.uniform() < 0.85:
                        grow(px, py, a + s * rng.uniform(0.35, 0.7), length * rng.uniform(0.55, 0.75),
                             width * 0.65, depth - 1)

        for i in range(n_dend):
            ang = axon_dir + math.pi + (i - (n_dend - 1) / 2) * (2 * math.pi * 0.72 / n_dend) + rng.normal(0, 0.12)
            sx = math.cos(ang) * soma_r * 0.85
            sy = math.sin(ang) * soma_r * 0.85
            grow(sx, sy, ang, rng.uniform(120, 170), rng.uniform(14, 18), 2)
        # Axón
        ca, sa = math.cos(axon_dir), math.sin(axon_dir)
        pts = []
        for i in range(13):
            u = i / 12
            d = soma_r * 0.8 + u * axon_len
            off = math.sin(u * math.pi * 1.6 + seed) * axon_curve * u
            pts.append((ca * d - sa * off, sa * d + ca * off))
        self.axon = catmull(pts, 4, closed=False)
        self.axon_w = 22
        self.terms = []
        if terminal:
            ex, ey = self.axon[-1]
            a0 = math.atan2(self.axon[-1][1] - self.axon[-6][1], self.axon[-1][0] - self.axon[-6][0])
            for s in (-0.7, 0.0, 0.7):
                l = rng.uniform(60, 90)
                mid = (ex + math.cos(a0 + s * 0.6) * l * 0.5, ey + math.sin(a0 + s * 0.6) * l * 0.5)
                end = (ex + math.cos(a0 + s) * l, ey + math.sin(a0 + s) * l)
                self.terms.append(([(ex, ey), mid, end], end))
        # Ovillo: fibras pre-generadas
        self.tangle = []
        for _ in range(70):
            ang = rng.uniform(0, 6.28)
            rr = soma_r * rng.uniform(0.45, 0.88)
            px, py = math.cos(ang) * rr, math.sin(ang) * rr
            b = rng.uniform(0, 6.28)
            pts = [(px, py)]
            for _ in range(5):
                b += rng.normal(0, 0.9)
                px += math.cos(b) * soma_r * 0.12
                py += math.sin(b) * soma_r * 0.12
                if math.hypot(px, py) > soma_r * 0.9:
                    break
                pts.append((px, py))
            if len(pts) > 2:
                self.tangle.append(pts)
        self.soma = blob(0, 0, soma_r, seed + 3, 0, 0.07)

    def W(self, p):
        return (self.x + p[0] * self.sc, self.y + p[1] * self.sc)

    def axon_world(self):
        return [self.W(p) for p in self.axon]

    def draw(self, cv, t, a=1.0, tangle=0.0, health=1.0, body=MINT, glow=0.0, show_nucleus=True,
             tint=None, frag=0.0, dark=0.0):
        sc = self.sc
        ink = mix(INK, DARK_D, dark)
        col = mix(body, GRAY, (1 - health) * 0.8)
        if tint:
            col = mix(col, tint[0], tint[1])
        if glow > 0:
            cv.glow(self.x, self.y, self.soma_r * sc * 2.6, ORANGE_L, a=glow)
        ow = 2.4

        def dendrite_lines(pass_ink):
            for pts, w in self.branches:
                wp = [self.W(p) for p in pts]
                if pass_ink:
                    cv.line(wp, ink, w * sc + 2 * ow, a)
                else:
                    cv.line(wp, col, w * sc, a)

        def axon_lines(pass_ink):
            ax = self.axon_world()
            n = len(ax)
            if frag > 0:
                # Axón fragmentado: tramos con huecos crecientes.
                chunk = 4
                for i in range(0, n - 1, chunk):
                    seg = ax[i:i + chunk + 1]
                    keep = int(len(seg) * (1 - frag * 0.6))
                    if keep >= 2:
                        cv.line(seg[:keep], ink if pass_ink else col,
                                (self.axon_w * sc + 2 * ow) if pass_ink else self.axon_w * sc, a)
            else:
                cv.line(ax, ink if pass_ink else col,
                        (self.axon_w * sc + 2 * ow) if pass_ink else self.axon_w * sc, a)
            for tp, end in self.terms:
                wtp = [self.W(p) for p in tp]
                cv.line(wtp, ink if pass_ink else col, (10 * sc + 2 * ow) if pass_ink else 10 * sc, a * (1 - frag))
                e = self.W(end)
                cv.circle(e[0], e[1], 15 * sc + (ow if pass_ink else 0), fill=ink if pass_ink else col,
                          outline=None, a=a * (1 - frag))

        dendrite_lines(True)
        axon_lines(True)
        soma_w = [self.W(p) for p in self.soma]
        cv.poly(soma_w, fill=None, outline=ink, w=ow * 2, a=a)
        dendrite_lines(False)
        axon_lines(False)
        cv.poly(soma_w, fill=col, outline=None, a=a)
        # Brillo suave del soma
        cv.ellipse(self.x - self.soma_r * sc * 0.35, self.y - self.soma_r * sc * 0.4,
                   self.soma_r * sc * 0.3, self.soma_r * sc * 0.18, fill=(255, 255, 255), outline=None, a=a * 0.25)
        if show_nucleus:
            ncol = mix(ORANGE, GRAY, (1 - health) * 0.6)
            nucleus(cv, self.x - 6 * sc, self.y + 4 * sc, self.soma_r * 0.42 * sc, seed=self.seed + 7, t=t, a=a,
                    col=ncol, cold=mix(ORANGE_D, GRAY, (1 - health) * 0.6))
        if tangle > 0:
            nt = int(len(self.tangle) * tangle)
            for pts in self.tangle[:nt]:
                wp = [self.W(p) for p in pts]
                cv.stroke(wp, TAU_BAD, 2.8 * sc + 0.6, ow=1.1 * sc + 0.4, a=a * 0.95, ink=ink)


# -------------------------------- Cerebro -----------------------------------
BRAIN_CTRL = [(-285, -20), (-265, -100), (-210, -165), (-130, -208), (-30, -225), (70, -215), (160, -180),
              (232, -120), (272, -50), (278, 15), (250, 70), (195, 98), (140, 100), (100, 122), (40, 150),
              (-30, 160), (-95, 146), (-140, 116), (-162, 82), (-190, 60), (-232, 50), (-270, 25)]
BRAIN = catmull(BRAIN_CTRL, 6)


def _brain_gyri():
    """Surcos: curvas suaves que no se cruzan (se rechazan las que quedan muy cerca)."""
    rng = np.random.default_rng(21)
    shrink = [(x * 0.9 - 5, y * 0.88 - 8) for x, y in BRAIN]
    curves, used = [], []
    tries = 0
    while len(curves) < 34 and tries < 4000:
        tries += 1
        x, y = rng.uniform(-260, 260), rng.uniform(-210, 150)
        if not inside((x, y), shrink):
            continue
        a = rng.uniform(0, 6.28)
        pts = [(x, y)]
        for _ in range(int(rng.integers(4, 8))):
            a += rng.normal(0, 0.55)
            x += math.cos(a) * 24
            y += math.sin(a) * 24
            if not inside((x, y), shrink) or any(math.dist((x, y), u) < 26 for u in used):
                break
            pts.append((x, y))
        if len(pts) >= 3:
            c = catmull(pts, 5, closed=False)
            curves.append(c)
            used.extend(c[::2])
    return curves


BRAIN_GYRI = _brain_gyri()
SYLVIAN = catmull([(-165, 70), (-100, 50), (-30, 30), (40, 10), (80, -10)], 5, closed=False)
CENTRAL = catmull([(20, -222), (10, -160), (-10, -100), (-15, -40), (-30, 10)], 5, closed=False)


def draw_brain(cv, t, a=1.0, col=PINK_L, dark=0.0, sparkle=0.0, spread=None):
    ink = mix(INK, DARK_D, dark)
    # Tronco y cerebelo (detrás)
    stem = [(95, 105), (140, 100), (150, 160), (140, 250), (100, 255), (90, 170)]
    cv.poly(catmull(stem, 5), fill=mix(col, (200, 140, 130), 0.3), outline=ink, w=2.8, a=a)
    cb = blob(190, 142, 78, 5, 0, 0.03, sx=1.1, sy=0.72)
    cv.poly(cb, fill=mix(col, (210, 150, 140), 0.25), outline=ink, w=2.8, a=a)
    for i in range(5):
        yy = 112 + i * 13
        cv.line(catmull([(125, yy + 6), (170, yy), (215, yy - 2), (262, yy + 6)], 4, closed=False),
                mix(ink, col, 0.4), w=1.8, a=a * 0.8)
    cv.poly(BRAIN, fill=col, outline=None, a=a)
    for c in BRAIN_GYRI:
        cv.line(c, mix(ink, col, 0.35), w=3.0, a=a * 0.9)
    cv.line(SYLVIAN, ink, w=3.2, a=a)
    cv.line(CENTRAL, mix(ink, col, 0.2), w=2.8, a=a)
    if spread is not None:
        spread(cv)
    cv.poly(BRAIN, fill=None, outline=ink, w=3.4, a=a)
    # brillo
    cv.ellipse(-120, -150, 90, 34, fill=(255, 255, 255), outline=None, a=a * 0.18)
    if sparkle > 0:
        rng = np.random.default_rng(4)
        for i in range(90):
            x, y = rng.uniform(-270, 270), rng.uniform(-215, 150)
            if not inside((x, y), BRAIN):
                continue
            ph = rng.uniform(0, 6.28)
            k = (math.sin(t * 3 + ph) * 0.5 + 0.5) * sparkle
            cv.circle(x, y, 4 + 3 * k, fill=(255, 250, 220), outline=None, a=0.9 * k)
            cv.glow(x, y, 16, (255, 230, 170), a=0.6 * k, steps=4)


# ----------------------------------------------------------------------------
# Escenas
# ----------------------------------------------------------------------------
class SceneInfo:
    def __init__(self, sc, data):
        self.start = sc["start"]
        self.end = sc["end"]
        self.D = self.end - self.start
        self.L = [l["start"] - self.start for l in sc["lines"]]
        self.E = [l["end"] - self.start for l in sc["lines"]]
        self.data = data

    def kw(self, li, word, which="say"):
        """Momento aproximado (local) en que la voz dice `word` en la frase li."""
        ln = self.data["lines"][li]
        txt = ln.get(which, ln["sub"]) if which == "say" else ln["sub"]
        pos = txt.lower().find(word.lower())
        if pos < 0:
            pos = 0
        return self.L[li] + (self.E[li] - self.L[li]) * pos / max(1, len(txt))


def cam_ease(t, keys):
    """keys: lista de (t, cx, cy, z). Interpola suavemente."""
    if t <= keys[0][0]:
        return keys[0][1:]
    for (t0, *a), (t1, *b) in zip(keys, keys[1:]):
        if t <= t1:
            k = smooth((t - t0) / (t1 - t0))
            return tuple(lerp(p, q, k) for p, q in zip(a, b))
    return keys[-1][1:]


# 1 --------------------------------------------------------------------------
def scene_intro(img, t, S):
    bg_gradient(img, (246, 240, 226), PAPER_D)
    L = S.L
    zoom_t = S.kw(1, "proteína tau")
    cx, cy, z = cam_ease(t, [(0, 0, -150, 0.8), (L[0] - 0.4, 0, -150, 0.8), (L[0] + 1.5, 0, -10, 1.0),
                             (L[1], 0, -10, 1.12), (zoom_t, -40, 30, 1.35),
                             (S.D, -60, 60, 3.2)])
    cv = Canvas(img, cx, cy, z)
    rings(cv, t, PAPER_D, a=0.9)
    rays(cv, t, (214, 200, 172), a=0.55, n=70)
    ba = ramp(t, 0.6, 2.2)
    draw_brain(cv, t, a=ba, sparkle=ramp(t, L[0] + 1.0, L[0] + 2.5) * (1 - ramp(t, S.D - 2, S.D)))
    label(cv, 150, -170, "86.000 millones de neuronas", t, S.kw(0, "ochenta"), dx=90, dy=-60, size=24,
          t_out=L[1] + 0.5)
    title(cv, "La proteína tau", 95, 76, ramp(t, 0.2, 1.0) * (1 - ramp(t, L[0] - 0.6, L[0] + 0.3)),
          sub="Cómo actúa en el cerebro", subsize=32)
    # Al final: la cámara "entra" al cerebro y aparece una neurona (continuidad con la escena 2).
    fin = ramp(t, zoom_t + 1.0, S.D)
    if fin > 0:
        cv.circle(-60, 60, 40 * fin + 1, fill=CREAM, outline=None, a=fin * 0.8)


# 2 --------------------------------------------------------------------------
NEURON_MAIN = Neuron(-420, 30, 1.0, seed=2, axon_len=900, soma_r=90, n_dend=7, axon_curve=60)


def scene_neurona(img, t, S):
    bg_gradient(img, (247, 242, 228), PAPER_D)
    L = S.L
    axon = NEURON_MAIN.axon_world()
    mid = path_point(axon, 0.45)
    cx, cy, z = cam_ease(t, [(0, -300, 40, 1.6), (1.5, -150, 30, 0.95), (L[1] + 1.5, 0, 30, 0.88),
                             (S.D - 2.0, mid[0], mid[1], 2.2), (S.D, mid[0], mid[1], 4.0)])
    cv = Canvas(img, cx, cy, z)
    rings(cv, t, PAPER_D, a=0.8, x=-420, y=30)
    rays(cv, t, (218, 206, 180), a=0.5, n=60, x=-420, y=30, r0=100)
    NEURON_MAIN.draw(cv, t)
    # Carga viajando por el axón (anterógrada) y alguna de regreso (retrógrada).
    cargo_a = ramp(t, L[1] - 0.3, L[1] + 0.8)
    if cargo_a > 0:
        for i in range(14):
            back = i % 5 == 4
            s = ((t - L[1]) * 0.07 + i / 14) % 1.0
            if back:
                s = 1 - s
            p = path_point(axon, s)
            kind = i % 3
            if kind == 0:
                vesicle(cv, p[0], p[1], 7, BLUE, a=cargo_a)
            elif kind == 1:
                mitochondrion(cv, p[0], p[1], 18, 9, 0.05, a=cargo_a)
            else:
                vesicle(cv, p[0], p[1], 6, PINK_L, a=cargo_a)
    term = NEURON_MAIN.terms[1][1]
    label(cv, -420, -40, "cuerpo celular", t, S.kw(0, "cuerpo"), dx=-40, dy=-130, t_out=L[1] + 1)
    label(cv, -426, 34, "núcleo", t, S.kw(0, "núcleo"), dx=-120, dy=110, t_out=L[1] + 1)
    ap = path_point(axon, 0.35)
    label(cv, ap[0], ap[1], "axón", t, S.kw(0, "axón"), dx=0, dy=-120, t_out=S.D - 2.5)
    label(cv, term[0], term[1], "sinapsis", t, S.kw(1, "sinapsis"), dx=-30, dy=-110, t_out=S.D - 2.5)


# 3 / 4 / 5 comparten el "interior del axón" --------------------------------
MTS = [Microtubule(-1400, 2600, -175, seed=1), Microtubule(-1400, 2600, 5, seed=2),
       Microtubule(-1400, 2600, 185, seed=3)]


def axon_interior_bg(img, cv, t, dark=0.0, seed=7):
    c1 = mix(CYTO, DARK, dark)
    c2 = mix(CYTO_D, DARK_D, dark)
    bg_gradient(img, c1, c2)
    rays(cv, t, mix((226, 240, 232), (200, 220, 210), dark), a=0.22 + 0.2 * dark, n=50)
    # Membrana del axón (arriba y abajo) como en la referencia (borde verde).
    for sgn in (-1, 1):
        yy = sgn * 330
        pts = [(cv.cx - 900 + i * 40, yy + math.sin(i * 0.4 + t * 0.4) * 6) for i in range(46)]
        cv.line(pts, mix(MINT_D, (60, 100, 100), dark), w=26, a=0.95)
        cv.line(pts, mix(MINT, (90, 130, 128), dark), w=14, a=0.95)
    particles(cv, t, seed, 26, 1400, [mix(BLUE, (110, 150, 160), dark), None, None], par=0.5, dark=dark)


def scene_microtubulos(img, t, S):
    L = S.L
    cx, cy, z = cam_ease(t, [(0, 0, 0, 0.62), (2.5, 60, 0, 0.9), (L[1], 150, 0, 1.0),
                             (S.D, 330, 10, 1.05)])
    cv = Canvas(img, cx, cy, z)
    axon_interior_bg(img, cv, t)
    for m in MTS:
        m.draw(cv, t)
    # Quinesina con vesícula (hacia la derecha) y otra con mitocondria.
    ka = ramp(t, L[1] - 0.4, L[1] + 0.6)
    kx = -250 + (t - L[1] + 1) * 60
    motor(cv, kx, MTS[1].y - Microtubule.R, t, "vesicle", KIN, a=ka)
    motor(cv, kx + 480, MTS[0].y - Microtubule.R, t + 0.3, "mito", (206, 150, 96), a=ka, speed_phase=0.4)
    label(cv, cx - 380, MTS[2].y, "microtúbulo", t, S.kw(0, "microtúbulos"), dx=0, dy=-80)
    label(cv, cx + 126, MTS[2].y - 6, "tubulina", t, S.kw(0, "tubulina"), dx=150, dy=-75)
    label(cv, kx + 10, MTS[1].y - 50, "quinesina", t, S.kw(1, "quinesina"), dx=-150, dy=40, size=24)
    label(cv, kx, MTS[1].y - 130, "carga", t, S.kw(1, "carga") - 0.4, dx=110, dy=-40, size=24)


TAU_SITES = []
for mi, m in enumerate(MTS):
    for j in range(-6, 20):
        side = -1 if (j + mi) % 2 == 0 else 1
        TAU_SITES.append((mi, -1300 + j * 230 + mi * 80, side, 100 + mi * 37 + j))


def tau_site_draw(cv, t, site, k_bound, phos=0.0, drift=(0, 0), a=1.0, col=TAU, dark=0.0):
    mi, x, side, seed = site
    m = MTS[mi]
    y = m.y + side * (Microtubule.R + 2)
    fx = x + drift[0]
    fy = y + side * 120 + drift[1]
    px = lerp(fx, x, k_bound)
    py = lerp(fy, y, k_bound)
    pts = tau_points(px, py, k_bound, t, seed, side=side)
    draw_tau(cv, pts, col=col, a=a, phos=phos, dark=dark)
    return pts


def scene_tau(img, t, S):
    L = S.L
    cx, cy, z = cam_ease(t, [(0, 330, 10, 1.05), (L[1], 420, 10, 1.1), (S.D, 470, 5, 1.45)])
    cv = Canvas(img, cx, cy, z)
    axon_interior_bg(img, cv, t)
    grow = ramp(t, L[1] + 1.5, S.D) * 8
    for m in MTS:
        m.draw(cv, t)
    # Vista: moléculas de tau llegan flotando y se unen.
    left = cx - W / 2 / z - 100
    right = cx + W / 2 / z + 100
    for i, site in enumerate(TAU_SITES):
        if not left < site[1] < right:
            continue
        t0 = L[0] + 0.8 + (i % 7) * 0.45
        k = ramp(t, t0, t0 + 1.8)
        a = ramp(t, t0 - 1.0, t0)
        tau_site_draw(cv, t, site, k, drift=(math.sin(i) * 80, math.cos(i) * 30), a=a)
    # Gen MAPT (icono de ADN)
    ga = ramp(t, S.kw(0, "gen") - 0.3, S.kw(0, "gen") + 0.4) * (1 - ramp(t, L[1] + 0.5, L[1] + 1.2))
    if ga > 0:
        gx, gy = cx - 400, cy - 250
        pts1 = [(gx - 90 + i * 6, gy + math.sin(i * 0.5 + t) * 18) for i in range(31)]
        pts2 = [(gx - 90 + i * 6, gy - math.sin(i * 0.5 + t) * 18) for i in range(31)]
        for i in range(0, 31, 3):
            cv.line([pts1[i], pts2[i]], INK, w=2, a=ga)
        cv.stroke(pts1, (236, 130, 110), 4, ow=1.6, a=ga)
        cv.stroke(pts2, (110, 160, 210), 4, ow=1.6, a=ga)
        label(cv, gx + 100, gy, "gen MAPT (cromosoma 17)", t, S.kw(0, "gen"), dx=150, dy=0, size=22, dot=False)
    lab_site = sorted([s for s in TAU_SITES if s[0] == 1], key=lambda s: abs(s[1] - (cx - 80)))[:1]
    if lab_site:
        s = lab_site[0]
        label(cv, s[1], MTS[1].y + s[2] * 36, "tau", t, S.kw(0, "tau") + 0.8, dx=40, dy=s[2] * 95, size=30,
              color=ORANGE_L)
    # Ensamblaje: dímeros nuevos llegando al extremo visible
    if grow > 0:
        for j in range(6):
            k = clamp(grow / 8 - j * 0.12)
            if k <= 0 or k >= 1:
                continue
            ex = cx + W / 2 / z - 150 + math.sin(j) * 60
            ey = MTS[1].y + (1 - k) * (-160 + j * 40)
            cv.rrect(ex - 5, ey - 5, ex + 5, ey + 5, 3, fill=TUB_A, outline=INK, w=1, a=1 - k)
    label(cv, cv.wx(W * 0.74), MTS[0].y + Microtubule.R, "estabiliza", t, S.kw(1, "estabiliza"), dx=0, dy=55,
          size=24, color=MINT_L)
    label(cv, cv.wx(W * 0.28), MTS[2].y - Microtubule.R, "favorece el ensamblaje", t, S.kw(1, "favorece"),
          dx=0, dy=-55, size=24, color=MINT_L)


def enzyme(cv, x, y, r, col, t, mouth=0.4, a=1.0, name=None):
    """Enzima tipo "pac-man" (dibujo simplificado)."""
    pts = []
    op = mouth * (0.6 + 0.4 * abs(math.sin(t * 3)))
    for i in range(40):
        ang = op + (2 * math.pi - 2 * op) * i / 39 + math.pi
        rr = r * (1 + 0.04 * math.sin(ang * 5 + t))
        pts.append((x + math.cos(ang) * rr, y + math.sin(ang) * rr))
    pts.append((x, y))
    cv.poly(pts, fill=col, outline=INK, w=2.5, a=a)
    cv.circle(x - r * 0.2, y - r * 0.45, r * 0.12, fill=INK, outline=None, a=a)
    cv.ellipse(x - r * 0.45, y - r * 0.1, r * 0.2, r * 0.12, fill=(255, 255, 255), outline=None, a=a * 0.3)
    if name:
        cv.text(x, y + r + 24, name, 20, INK, a=a)


def scene_fosforilacion(img, t, S):
    L, E = S.L, S.E
    cx, cy, z = cam_ease(t, [(0, 80, -100, 1.4), (L[1], 80, -110, 1.7), (S.D, 70, -110, 1.8)])
    cv = Canvas(img, cx, cy, z)
    axon_interior_bg(img, cv, t, seed=9)
    MTS[1].draw(cv, t)
    # Tres taus sobre el microtúbulo del medio (lado superior).
    A = (1, 80, -1, 501)
    B = (1, -140, -1, 502)
    C = (1, 300, -1, 503)
    # Guion de eventos de la tau A
    add1 = L[0] + 2.0
    add2 = S.kw(1, "con más") + 0.2
    add3 = add2 + 0.6
    rem_start = L[2] + 1.2
    phos = (ramp(t, add1, add1 + 0.4) + ramp(t, add2, add2 + 0.4) + ramp(t, add3, add3 + 0.4)
            - ramp(t, rem_start, rem_start + 0.5) - ramp(t, rem_start + 1.1, rem_start + 1.6))
    detach = ramp(t, add3 + 0.2, add3 + 1.1) * (1 - ramp(t, rem_start + 1.8, rem_start + 3.2))
    for site, ph in ((B, 1), (C, 1)):
        tau_site_draw(cv, t, site, 1.0, phos=ph * ramp(t, L[0] + 3.5, L[0] + 4), drift=(0, 0))
    tau_site_draw(cv, t, A, 1 - detach, phos=phos, drift=(40, -60))
    # Quinasa: se acerca a la tau A en los momentos de fosforilación.
    ka = ramp(t, L[0] + 0.3, L[0] + 1.0) * (1 - ramp(t, L[2] - 0.2, L[2] + 0.5))
    kpos_rest = (-170, -245)
    near = max(ramp(t, add1 - 0.8, add1) * (1 - ramp(t, add1 + 0.4, add1 + 1.2)),
               ramp(t, add2 - 0.8, add2) * (1 - ramp(t, add3 + 0.4, add3 + 1.2)))
    target = (A[1] + 60 + 40 * detach, MTS[1].y - 150 - 60 * detach)
    kx = lerp(kpos_rest[0], target[0], near)
    ky = lerp(kpos_rest[1], target[1], near)
    enzyme(cv, kx, ky, 38, KINASE, t, a=ka)
    label(cv, kx, ky - 38, "quinasa (agrega fosfato)", t, S.kw(0, "quinasas"), dx=-60, dy=-60, size=22,
          t_out=L[2] - 0.3)
    # Fosfatasa: aparece en la tercera frase y los quita.
    pa = ramp(t, S.kw(0, "fosfatasas") - 0.3, S.kw(0, "fosfatasas") + 0.4)
    prest = (330, -215)
    pnear = ramp(t, rem_start - 0.8, rem_start) * (1 - ramp(t, rem_start + 2.0, rem_start + 2.8))
    ptarget = (A[1] + 100 + 40 * detach, MTS[1].y - 160 - 60 * detach)
    px = lerp(prest[0], ptarget[0], pnear)
    py = lerp(prest[1], ptarget[1], pnear)
    enzyme(cv, px, py, 38, PHOSPHATASE, t + 1, a=pa)
    label(cv, px, py - 38, "fosfatasa (quita fosfato)", t, S.kw(0, "fosfatasas"), dx=-30, dy=-50, size=22,
          t_out=S.kw(2, "equilibrio") - 0.5)
    label(cv, A[1] + 40, MTS[1].y - 200, "se suelta", t, S.kw(1, "se suelta"), dx=0, dy=0, size=24,
          t_out=rem_start + 1.2, color=ORANGE_L)
    label(cv, cx, cy - 165, "equilibrio dinámico", t, S.kw(2, "equilibrio"), dx=0, dy=0, size=26, color=MINT_L)


# 6 --------------------------------------------------------------------------
def scene_patologia(img, t, S):
    L = S.L
    dark = ramp(t, 0.5, L[0] + 3.0) * 0.85
    cx, cy, z = cam_ease(t, [(0, 200, 0, 1.0), (L[1], 240, 0, 0.95), (L[2], 280, 0, 0.92), (S.D, 300, 0, 1.0)])
    cv = Canvas(img, cx, cy, z)
    axon_interior_bg(img, cv, t, dark=dark, seed=13)
    destroy = ramp(t, L[1] + 0.2, L[2] + 1.5)
    for m in MTS:
        m.draw(cv, t, destroy=destroy, dark=dark)
    left = cx - W / 2 / z - 100
    right = cx + W / 2 / z + 100
    hyper = ramp(t, S.kw(0, "hiperfosforila") - 0.5, L[1])
    det = ramp(t, L[1] - 0.2, L[1] + 2.0)
    for i, site in enumerate(TAU_SITES):
        if not left < site[1] < right:
            continue
        ph = 1 + hyper * (2 + (i % 2))
        k = 1 - clamp(det * 1.4 - (i % 4) * 0.12)
        draw_col = mix(TAU, TAU_BAD, hyper * 0.6)
        tau_site_draw(cv, t, site, k, phos=ph, drift=(math.sin(i * 2.1) * 60, math.cos(i) * 40 + 30),
                      col=draw_col, dark=dark)
    # Carga detenida: la quinesina avanza y se detiene cuando el riel se rompe.
    stall = ramp(t, L[2] - 0.5, L[2] + 0.5)
    kx = 150 + min(t, L[2]) * 30
    fall = ramp(t, L[2] + 0.5, S.D)
    motor(cv, kx, MTS[1].y - Microtubule.R - fall * 60, t * (1 - stall) + L[2] * stall, "vesicle",
          mix(KIN, GRAY, stall * 0.6), a=1 - fall * 0.5, stall=stall)
    if stall > 0:
        # Señal de "alto": X roja sobre la carga
        xx, yy = kx + 60, MTS[1].y - 160 - fall * 60
        cv.circle(xx, yy, 22, fill=(214, 84, 70), outline=INK, w=2.5, a=stall)
        cv.line([(xx - 9, yy - 9), (xx + 9, yy + 9)], CREAM, w=4, a=stall)
        cv.line([(xx + 9, yy - 9), (xx - 9, yy + 9)], CREAM, w=4, a=stall)
    rays(cv, t, (255, 230, 190), a=0.18 * dark, n=40)
    label(cv, cx - 250, MTS[0].y - 60, "hiperfosforilación", t, S.kw(0, "hiperfosforila"), dx=0, dy=-70,
          size=26, color=(248, 214, 150), t_out=L[1] + 1.5)
    label(cv, cx + 200, MTS[2].y, "microtúbulos inestables", t, S.kw(1, "inestables"), dx=0, dy=90, size=24,
          t_out=L[2] + 1.0)
    label(cv, kx + 60, MTS[1].y - 180, "transporte interrumpido", t, S.kw(2, "interrumpe"), dx=0, dy=-70,
          size=24, color=(248, 214, 150))


# 7 --------------------------------------------------------------------------
NEURON_TANGLE = Neuron(0, 0, 1.0, seed=5, axon_len=700, soma_r=150, n_dend=7, axon_curve=40)


def scene_agregacion(img, t, S):
    L = S.L
    tangle_t = S.kw(2, "finalmente")
    fil_t = S.kw(2, "filamentos")
    oligo_t = S.kw(1, "oligómeros")
    # Vista molecular hasta "finalmente"; luego se aleja hacia la neurona.
    k_view = ramp(t, tangle_t - 0.3, tangle_t + 1.2)
    img_mol = img if k_view < 1 else None
    if img_mol is not None:
        cv = Canvas(img, 0, 0, lerp(1.0, 1.25, t / S.D))
        c1, c2 = DARK, DARK_D
        bg_gradient(img, mix(c1, (80, 100, 104), 0.2), c2)
        rays(cv, t, (240, 210, 170), a=0.22, n=60)
        particles(cv, t, 17, 20, 900, [(90, 120, 128), None], par=0.3, dark=1)
        rng = np.random.default_rng(31)
        N = 30
        home = [(rng.uniform(-560, 560), rng.uniform(-300, 300)) for _ in range(N)]
        groups = [(-330, -120), (-40, 170), (260, -150), (430, 150), (-470, 180)]
        misfold = ramp(t, L[0] + 0.5, L[0] + 3.0)
        clump = ramp(t, L[0] + 2.0, oligo_t + 0.5)
        fil = ramp(t, fil_t - 0.5, fil_t + 2.0)
        for i, (hx, hy) in enumerate(home):
            g = groups[i % len(groups)]
            ang = i * 2.4
            tx = g[0] + math.cos(ang) * 38
            ty = g[1] + math.sin(ang) * 30
            x = lerp(hx, tx, clump) + math.sin(t * 0.6 + i) * 12 * (1 - clump)
            y = lerp(hy, ty, clump) + math.cos(t * 0.5 + i) * 10 * (1 - clump)
            x = lerp(x, 0, fil * 0.3)
            a = 1 - fil
            if a <= 0.02:
                continue
            pts = tau_points(x, y, 0.0, t * (1 - misfold * 0.6), seed=900 + i)
            # Mal plegada: más compacta
            pts = [(x + (px - x) * (1 - 0.45 * misfold), y + (py - y) * (1 - 0.45 * misfold)) for px, py in pts]
            draw_tau(cv, pts, col=mix(TAU, TAU_BAD, misfold), a=a, phos=3 * (1 - misfold * 0.4), pk=1, dark=1)
        if clump > 0.5:
            for g in groups:
                cv.glow(g[0], g[1], 120, (230, 90, 60), a=0.35 * (clump - 0.5) * 2 * (1 - fil))
        # Filamentos helicoidales apareados
        if fil > 0:
            for fi, (fy, ph) in enumerate(((-170, 0.0), (0, 1.3), (170, 2.6))):
                length = 900 * fil
                x0 = -length / 2
                s1, s2 = [], []
                for i in range(90):
                    u = i / 89
                    x = x0 + u * length
                    s1.append((x, fy + math.sin(u * 22 + ph) * 22))
                    s2.append((x, fy - math.sin(u * 22 + ph) * 22))
                cv.stroke(s2, mix(TAU_BAD, INK, 0.25), 9, ow=2.2, a=fil)
                cv.stroke(s1, TAU_BAD, 9, ow=2.2, a=fil)
        label(cv, groups[1][0], groups[1][1] - 40, "oligómeros", t, oligo_t, dx=0, dy=-90, size=26,
              color=(248, 214, 150), t_out=fil_t - 0.4)
        label(cv, 0, -170, "filamentos helicoidales apareados", t, fil_t, dx=0, dy=-75, size=24,
              color=(248, 214, 150), t_out=tangle_t)
        label(cv, -380, 250, "tau mal plegada", t, L[0] + 1.5, dx=0, dy=0, size=24, color=(248, 214, 150),
              t_out=oligo_t - 0.5)
    if k_view > 0:
        layer = Image.new("RGB", img.size)
        z = lerp(4.0, 1.25, smooth(k_view)) * lerp(1, 1.08, ramp(t, tangle_t, S.D))
        cv2 = Canvas(layer, 20, 0, z)
        bg_gradient(layer, (120, 140, 136), DARK_D)
        rays(cv2, t, (255, 220, 170), a=0.3, n=60)
        NEURON_TANGLE.draw(cv2, t, tangle=ramp(t, tangle_t + 0.3, tangle_t + 2.5), dark=0.2,
                           tint=((170, 190, 170), 0.2))
        label(cv2, 90, -60, "ovillo neurofibrilar", t, S.kw(2, "ovillos") + 0.2, dx=180, dy=-120, size=26,
              color=(248, 214, 150))
        if k_view >= 1:
            img.paste(layer)
        else:
            img.paste(Image.blend(img, layer, k_view))


# 8 --------------------------------------------------------------------------
NET = [Neuron(-560, -80, 0.42, seed=40, axon_len=560, axon_dir=0.25, soma_r=90, axon_curve=30),
       Neuron(-150, 60, 0.42, seed=41, axon_len=560, axon_dir=-0.35, soma_r=90, axon_curve=30),
       Neuron(240, -110, 0.42, seed=42, axon_len=560, axon_dir=0.35, soma_r=90, axon_curve=30),
       Neuron(620, 60, 0.42, seed=43, axon_len=400, axon_dir=0.0, soma_r=90, axon_curve=30)]


def scene_propagacion(img, t, S):
    L = S.L
    brain_t = L[2] - 0.3
    k_brain = ramp(t, brain_t - 0.4, brain_t + 0.8)
    if k_brain < 1:
        # Parte A: neurona muere; parte B: red de neuronas y propagación.
        net_t = L[1] - 0.2
        k_net = ramp(t, net_t - 0.4, net_t + 0.6)
        base = Image.new("RGB", img.size)
        if k_net < 1:
            cv = Canvas(base, 20, 0, 1.3 - 0.1 * ramp(t, 0, L[1]))
            bg_gradient(base, (120, 140, 136), DARK_D)
            rays(cv, t, (255, 220, 170), a=0.3 * (1 - ramp(t, 1, L[1])), n=60)
            die = ramp(t, S.kw(0, "neurona muere") - 0.5, S.kw(0, "neurona muere") + 1.5)
            NEURON_TANGLE.draw(cv, t, tangle=1.0, dark=0.2, health=1 - die * 0.9, frag=die,
                               tint=((170, 190, 170), 0.2), a=1 - die * 0.35)
            label(cv, -60, -160, "sinapsis que fallan", t, S.kw(0, "sinapsis"), dx=-120, dy=-110, size=24,
                  color=(248, 214, 150))
        if k_net > 0:
            layer = Image.new("RGB", img.size)
            cv2 = Canvas(layer, 30, 0, 1.0 + 0.05 * ramp(t, net_t, brain_t))
            bg_gradient(layer, (104, 128, 126), DARK_D)
            rays(cv2, t, (255, 220, 170), a=0.25, n=60)
            lt = t - net_t
            for i, n in enumerate(NET):
                inf = ramp(lt, 0.8 + i * 1.7, 1.9 + i * 1.7)
                n.draw(cv2, t, tangle=inf * 0.8, dark=0.5, glow=0.35 * inf, tint=((230, 160, 110), inf * 0.35))
                if i < len(NET) - 1:
                    ax = n.axon_world()
                    for j in range(5):
                        s = clamp((lt - 1.0 - i * 1.7) / 1.5 - j * 0.12)
                        if 0 < s < 1:
                            p = path_point(ax, s)
                            cv2.glow(p[0], p[1], 30, (255, 150, 90), a=0.6, steps=5)
                            cv2.circle(p[0], p[1], 6, fill=TAU_BAD, outline=INK, w=1.6)
            label(cv2, NET[1].x, NET[1].y, "propagación tipo prion", t, S.kw(1, "prion") - 0.3, dx=0, dy=-170,
                  size=26, color=(248, 214, 150))
            base = layer if k_net >= 1 else Image.blend(base, layer, k_net)
        img.paste(base)
    if k_brain > 0:
        layer = Image.new("RGB", img.size)
        ent_t = S.kw(2, "corteza entorrinal")
        hip_t = S.kw(2, "hipocampo")
        ctx_t = S.kw(2, "corteza cerebral")
        cv3 = Canvas(layer, -10, -10, lerp(1.35, 1.2, ramp(t, brain_t, S.D)))
        bg_gradient(layer, (70, 92, 100), DARK_D)
        rays(cv3, t, (255, 220, 170), a=0.25, n=70)

        def spread(cv):
            e = ramp(t, ent_t - 0.3, ent_t + 1.0)
            h = ramp(t, hip_t - 0.3, hip_t + 1.2)
            c = ramp(t, ctx_t - 0.3, S.D - 0.5)
            if c > 0:
                # Mancha que se extiende por la corteza, recortada al contorno.
                mask = Image.new("L", layer.size, 0)
                md = ImageDraw.Draw(mask)
                md.polygon([cv.P(x, y) for x, y in BRAIN], fill=255)
                gl = layer.copy()
                gcv = Canvas(gl, cv.cx, cv.cy, cv.z)
                for k in range(12):
                    rr = 60 + c * 380 * (1 - k / 12)
                    gcv.circle(-40, 110, rr, fill=(226, 110, 60), outline=None, a=0.08)
                layer.paste(gl, (0, 0), mask)
            if h > 0:
                hp = catmull([(-110, 92), (-60, 80), (0, 72), (45, 60), (60, 40)], 5, closed=False)
                cv.line(hp, (226, 110, 60), w=26, a=0.35 * h)
                cv.line(hp, (240, 150, 80), w=14, a=0.9 * h)
            if e > 0:
                cv.glow(-110, 118, 70, (240, 110, 60), a=0.9 * e)
                cv.circle(-110, 118, 16, fill=(226, 96, 58), outline=INK, w=2, a=e)

        draw_brain(cv3, t, a=1.0, col=(226, 176, 166), dark=0.3, spread=spread)
        label(cv3, -110, 118, "corteza entorrinal", t, ent_t, dx=-190, dy=30, size=22, color=(248, 214, 150))
        label(cv3, 20, 66, "hipocampo", t, hip_t, dx=230, dy=95, size=22, color=(248, 214, 150))
        label(cv3, 120, -150, "corteza cerebral", t, ctx_t, dx=120, dy=-80, size=22, color=(248, 214, 150))
        label(cv3, 0, 0, "Estadios de Braak (simplificado)", t, ent_t - 1.0, dx=0, dy=-300, size=20, dot=False,
              color=CREAM)
        if k_brain >= 1:
            img.paste(layer)
        else:
            img.paste(Image.blend(img, layer, k_brain))


# 9 --------------------------------------------------------------------------
NEURON_AD = Neuron(-120, 20, 0.95, seed=8, axon_len=700, soma_r=120, n_dend=7, axon_curve=50)


def scene_tauopatias(img, t, S):
    L = S.L
    bg_gradient(img, (247, 242, 228), PAPER_D)
    cv = Canvas(img, 0, 0, 1.0)
    rings(cv, t, PAPER_D, a=0.8)
    rays(cv, t, (214, 200, 172), a=0.5, n=60)
    k1 = ramp(t, L[1] - 0.4, L[1] + 0.4)
    k2 = ramp(t, L[2] - 0.5, L[2] + 0.3)
    title(cv, "Tauopatías", 300 - 60 * k1, 80, ramp(t, 0.3, 1.2) * (1 - k1),
          sub="enfermedades con acumulación anormal de tau", subsize=28)
    if k1 > 0 and k2 < 1:
        a = k1 * (1 - k2)
        cz = Canvas(img, 60, 30, 0.85)
        NEURON_AD.draw(cz, t, a=a, tangle=ramp(t, L[1] + 0.2, L[1] + 2.0))
        rng = np.random.default_rng(50)
        pa = ramp(t, S.kw(1, "placas") - 0.3, S.kw(1, "placas") + 0.8) * a
        for i, (px, py) in enumerate([(300, -220), (430, 150), (-460, 230), (-520, -200), (180, 260),
                                      (560, -120)]):
            r = rng.uniform(40, 60)
            cz.poly(blob(px, py, r, 60 + i, t * 0.3, 0.14), fill=PLAQUE, outline=INK, w=2.5, a=pa)
            for j in range(10):
                ang = rng.uniform(0, 6.28)
                rr = r * rng.uniform(0.1, 0.7)
                cz.circle(px + math.cos(ang) * rr, py + math.sin(ang) * rr, rng.uniform(3, 7),
                          fill=(120, 84, 52), outline=None, a=pa)
        label(cz, -60, -40, "ovillos de tau (dentro de la neurona)", t, S.kw(1, "ovillos"), dx=-60, dy=-230,
              size=24, color=ORANGE_L)
        label(cz, 300, -220, "placas de beta-amiloide (fuera)", t, S.kw(1, "placas"), dx=120, dy=-90,
              size=24, color=(222, 196, 160))
    if k2 > 0:
        cards = [("Alzheimer", S.L[2] - 0.2), ("Demencia frontotemporal", S.kw(2, "demencia")),
                 ("Parálisis supranuclear progresiva", S.kw(2, "parálisis")),
                 ("Encefalopatía traumática crónica", S.kw(2, "encefalopatía"))]
        for i, (txt, tin) in enumerate(cards):
            k = ramp(t, tin - 0.2, tin + 0.5)
            if k <= 0:
                continue
            y = -170 + i * 105
            x = lerp(-900, 0, smooth(k))
            cv.rrect(x - 330 + 4, y - 38 + 5, x + 330 + 4, y + 38 + 5, 20, fill=INK, outline=None, a=0.25 * k)
            cv.rrect(x - 330, y - 38, x + 330, y + 38, 20, fill=[ORANGE_L, MINT_L, PINK_L, (200, 220, 236)][i],
                     outline=INK, w=2.6, a=k)
            cv.circle(x - 290, y, 16, fill=TAU_BAD, outline=INK, w=2.2, a=k)
            cv.text(x + 15, y, txt, 30, INK, a=k)


# 10 -------------------------------------------------------------------------
NEURON_HOPE = Neuron(0, -10, 0.8, seed=12, axon_len=700, soma_r=110, n_dend=8, axon_curve=40)


def stars(cv, t, n=160, seed=3):
    rng = np.random.default_rng(seed)
    for i in range(n):
        x, y = rng.uniform(0, W * SS), rng.uniform(0, H * SS)
        r = rng.uniform(0.6, 2.4) * SS
        ph = rng.uniform(0, 6.28)
        a = 0.35 + 0.55 * (0.5 + 0.5 * math.sin(t * rng.uniform(0.8, 2.5) + ph))
        cv.d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 250, 235, int(255 * a)))
        if r > 1.9 * SS:
            cv.d.line([(x - 4 * r, y), (x + 4 * r, y)], fill=(255, 250, 235, int(120 * a)), width=SS)
            cv.d.line([(x, y - 4 * r), (x, y + 4 * r)], fill=(255, 250, 235, int(120 * a)), width=SS)


def antibody(cv, x, y, s, a):
    arms = [((x, y + 30 * s), (x, y)), ((x, y), (x - 26 * s, y - 30 * s)), ((x, y), (x + 26 * s, y - 30 * s))]
    for p, q in arms:
        cv.stroke([p, q], (180, 200, 240), 9 * s, ow=2, a=a)


def scene_futuro(img, t, S):
    L = S.L
    bg_gradient(img, (40, 62, 92), SPACE)
    cv = Canvas(img, 0, 0, lerp(1.0, 0.82, ramp(t, 0, S.D)))
    stars(cv, t)
    # Nebulosa rosada suave, como en el final de la referencia.
    cv.glow(-420, 200, 380, (190, 120, 170), a=0.35)
    cv.glow(480, -220, 300, (110, 150, 200), a=0.3)
    rays(cv, t, (255, 236, 200), a=0.35, n=80, x=0, y=-10, r0=120, length=900)
    healthy = ramp(t, L[1] - 0.5, L[1] + 2.0)
    NEURON_HOPE.draw(cv, t, glow=0.25 + 0.35 * healthy, tangle=0.6 * (1 - healthy))
    items = [(-390, -230, "Anticuerpos contra tau", S.kw(0, "anticuerpos"), "ab"),
             (420, -230, "Inhibidores de la agregación", S.kw(0, "fármacos"), "agg"),
             (430, 170, "Terapias que reducen su producción", S.kw(0, "terapias"), "aso")]
    end_t = S.E[-1] + 0.6
    for x, y, txt, tin, kind in items:
        k = ramp(t, tin - 0.2, tin + 0.6) * (1 - ramp(t, end_t, end_t + 0.8))
        if k <= 0:
            continue
        cv.circle(x, y - 5, 46, fill=(38, 60, 92), outline=(230, 220, 200), w=2.5, a=k)
        if kind == "ab":
            antibody(cv, x, y - 2, 1.0, k)
        elif kind == "agg":
            for j in range(5):
                cv.circle(x - 14 + (j % 3) * 14, y - 14 + (j // 3) * 16, 9, fill=TAU_BAD, outline=INK, w=1.6, a=k)
            cv.circle(x, y - 5, 34, fill=None, outline=(236, 96, 80), w=4, a=k)
            cv.line([(x - 24, y + 19), (x + 24, y - 29)], (236, 96, 80), w=4, a=k)
        else:
            pts = [(x - 30 + i * 3, y - 5 + math.sin(i * 0.6 + t * 2) * 10) for i in range(21)]
            cv.stroke(pts, (150, 200, 240), 4, ow=1.6, a=k)
            cv.line([(x - 4, y - 28), (x + 8, y + 18)], (240, 220, 120), w=3, a=k)
        f = font("Fredoka", 26 * SS * cv.z, "SemiBold")
        px, py = cv.P(x, y + 70)
        cv.d.text((px, py), txt, font=f, fill=rgba(CREAM, k), anchor="mm")
    # Título final
    ft = ramp(t, end_t + 0.4, end_t + 1.4)
    if ft > 0:
        ov = Canvas(img)
        ov.d.rectangle([0, 0, W * SS, H * SS], fill=rgba(SPACE, ft * 0.55))
        title(ov, "La proteína tau", 320, 78, ft, color=CREAM, shadow=False)
        f2 = font("Nunito", 24 * SS, "SemiBold")
        ov.d.text((W * SS / 2, 395 * SS), "Video educativo · contenido simplificado con fines de divulgación",
                  font=f2, fill=rgba(CREAM, ft * 0.85), anchor="mm")


SCENE_FUNCS = {
    "intro": scene_intro, "neurona": scene_neurona, "microtubulos": scene_microtubulos, "tau": scene_tau,
    "fosforilacion": scene_fosforilacion, "patologia": scene_patologia, "agregacion": scene_agregacion,
    "propagacion": scene_propagacion, "tauopatias": scene_tauopatias, "futuro": scene_futuro,
}


# ----------------------------------------------------------------------------
# Composición de cuadros
# ----------------------------------------------------------------------------
TL = None
INFOS = None


def load(build):
    global TL, INFOS
    with open(os.path.join(build, "timeline.json")) as f:
        TL = json.load(f)
    INFOS = [SceneInfo(sc, SCENES[i]) for i, sc in enumerate(TL["scenes"])]


def render_scene(i, t):
    img = Image.new("RGB", (W * SS, H * SS), PAPER)
    S = INFOS[i]
    SCENE_FUNCS[TL["scenes"][i]["id"]](img, t - S.start, S)
    return img


def wrap(txt, f, maxw, d):
    words = txt.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if d.textlength(test, font=f) > maxw and cur:
            lines.append(cur)
            cur = w
        else:
            cur = test
    lines.append(cur)
    # Balancear dos líneas
    if len(lines) == 2:
        best = None
        ws = txt.split()
        for k in range(1, len(ws)):
            a, b = " ".join(ws[:k]), " ".join(ws[k:])
            la, lb = d.textlength(a, font=f), d.textlength(b, font=f)
            if la <= maxw and lb <= maxw:
                sc = max(la, lb)
                if best is None or sc < best[0]:
                    best = (sc, [a, b])
        if best:
            lines = best[1]
    return lines


def draw_subs(im, t):
    """Un solo subtítulo por vez; solo hay fundido cuando hay un silencio entre bloques."""
    subs = TL["subs"]
    d = ImageDraw.Draw(im, "RGBA")
    for k, s in enumerate(subs):
        prev_end = subs[k - 1]["end"] if k > 0 else -1e9
        next_start = subs[k + 1]["start"] if k + 1 < len(subs) else 1e9
        fade_in = s["start"] - prev_end > 0.15
        fade_out = next_start - s["end"] > 0.15
        t0 = s["start"] - (0.08 if fade_in else 0)
        t1 = s["end"] + (0.12 if fade_out else 0)
        if not (t0 <= t < t1):
            continue
        a = 1.0
        if fade_in:
            a = min(a, ramp(t, s["start"] - 0.08, s["start"] + 0.06))
        if fade_out:
            a = min(a, 1 - ramp(t, s["end"], s["end"] + 0.12))
        f = font("Nunito", 31, "Bold")
        lines = wrap(s["text"], f, 1060, d)
        lh = 40
        tw = max(d.textlength(l, font=f) for l in lines)
        y0 = H - 40 - lh * len(lines)
        d.rounded_rectangle([W / 2 - tw / 2 - 22, y0 - 12, W / 2 + tw / 2 + 22, y0 + lh * len(lines) + 8],
                            radius=14, fill=(30, 22, 18, int(175 * a)))
        for j, l in enumerate(lines):
            d.text((W / 2, y0 + j * lh + lh / 2), l, font=f, fill=(252, 246, 230, int(255 * a)), anchor="mm")
        break


def frame(t):
    n = len(INFOS)
    i = 0
    while i < n - 1 and t >= INFOS[i].end:
        i += 1
    img = render_scene(i, t)
    # Fundido cruzado con la escena vecina alrededor del límite.
    if i < n - 1 and t > INFOS[i].end - XFADE / 2:
        k = smooth((t - (INFOS[i].end - XFADE / 2)) / XFADE)
        img = Image.blend(img, render_scene(i + 1, t), k)
    elif i > 0 and t < INFOS[i].start + XFADE / 2:
        k = smooth((t - (INFOS[i].start - XFADE / 2)) / XFADE)
        img = Image.blend(render_scene(i - 1, t), img, k)
    im = img.convert("RGB").resize((W, H), Image.LANCZOS)
    arr = np.asarray(im, np.float32) * post_map()
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    # Fundido desde/hacia negro al principio y al final
    fade = min(ramp(t, 0, 0.8), 1 - ramp(t, TL["duration"] - 1.2, TL["duration"]))
    if fade < 1:
        im = Image.blend(Image.new("RGB", (W, H), (0, 0, 0)), im, fade)
    draw_subs(im, t)
    return im


def render_chunk(args):
    build, idx, f0, f1, out = args
    load(build)
    p = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                          "-crf", "25", "-tune", "animation", "-pix_fmt", "yuv420p", out], stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        p.stdin.write(frame(fi / FPS).tobytes())
        if (fi - f0) % 150 == 0:
            print(f"  [{idx}] {fi - f0}/{f1 - f0}", flush=True)
    p.stdin.close()
    p.wait()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="build")
    ap.add_argument("--out", default="tau.mp4")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--frames", type=str, default=None, help="tiempos separados por coma para previsualizar")
    ap.add_argument("--chunk", type=float, default=12.0)
    args = ap.parse_args()
    load(args.build)
    if args.frames:
        for tt in args.frames.split(","):
            frame(float(tt)).save(os.path.join(args.build, f"preview_{float(tt):07.2f}.png"))
        return
    total = int(math.ceil(TL["duration"] * FPS))
    step = int(args.chunk * FPS)
    jobs = []
    for k, f0 in enumerate(range(0, total, step)):
        jobs.append((args.build, k, f0, min(total, f0 + step), os.path.join(args.build, f"seg_{k:03d}.mp4")))
    with Pool(args.workers) as pool:
        outs = pool.map(render_chunk, jobs, chunksize=1)
    lst = os.path.join(args.build, "segs.txt")
    with open(lst, "w") as f:
        for o in outs:
            f.write(f"file '{os.path.abspath(o)}'\n")
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-i", os.path.join(args.build, "mix.wav"), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                    "-shortest", "-movflags", "+faststart", args.out], check=True)
    for o in outs:
        os.remove(o)
    print("Listo:", args.out)


if __name__ == "__main__":
    main()
