// ASCII topographic terrain, infinite in every direction.
//
// The height at any point is a blend of two noise fields:
//   h = (1 - ridge) * smooth + ridge * ridged  (+ peaks, folded into both)
// Because that blend is linear in `ridge`, both fields are sampled on the CPU and
// the shader mixes them every frame, so ridges and sea level can animate for free.
//
// Everything is a pure function of world position: noise, domain warp, and
// peaks (hashed per tile). A FieldWindow holds the samples for the part of the
// world on screen and slides over it, computing only the newly visible strip.

export type TopoSettings = {
  seed: number;
  size: number; // feature size, in cells
  oct: number;
  rough: number;
  warp: number;
  peakChance: number; // chance that a peak tile holds a peak, 0..1
  peakTile: number; // peak tile size, in cells
  peakH: number;
  peakW: number;
  levels: number;
  index: number;
  maxE: number;
};

/** How many summits to label in a view this many cells wide. */
export const labelCount = (cols: number) => Math.max(3, Math.round(cols / 50));

export const defaultTopoSettings = (): TopoSettings => ({
  seed: Math.floor(Math.random() * 99999),
  size: 60,
  oct: 4,
  rough: 50,
  warp: 25,
  peakChance: 0.55,
  peakTile: 80,
  peakH: 80,
  peakW: 18,
  levels: 12,
  index: 5,
  maxE: 3200,
});

const NORM_STEPS = 32;

function rng(s: number) {
  return () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Integer hash -> 0..1, for per-tile peaks
function hash3(x: number, y: number, z: number) {
  let h =
    Math.imul(x, 374761393) +
    Math.imul(y, 668265263) +
    Math.imul(z, 2147483647);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function valueNoise(seed: number) {
  const r = rng(seed);
  const v = new Float32Array(256);
  const p = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    v[i] = r();
    p[i] = i;
  }
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  const L = (i: number, j: number) => v[p[(p[i & 255] + j) & 255]];
  const f = (t: number) => t * t * t * (t * (t * 6 - 15) + 10);
  return (x: number, y: number) => {
    const xi = Math.floor(x);
    const yi = Math.floor(y);
    const u = f(x - xi);
    const w = f(y - yi);
    const a = L(xi, yi);
    const b = L(xi + 1, yi);
    const c = L(xi, yi + 1);
    const d = L(xi + 1, yi + 1);
    return a + (b - a) * u + (c - a) * w + (a - b - c + d) * u * w;
  };
}

export type TopoWorld = {
  settings: TopoSettings;
  /** Writes [smooth, ridged] for a sample point, in braille-dot units (2 per cell across, 4 down). */
  sample: (wx: number, wy: number, out: Float32Array, i: number) => void;
  lo: number; // packing range
  hi: number;
  norm: { min: number; max: number }[]; // normalisation of the blend, per ridge step
};

export function createWorld(S: TopoSettings): TopoWorld {
  const n = valueNoise(S.seed);
  const nw1 = valueNoise(S.seed + 101);
  const nw2 = valueNoise(S.seed + 202);
  const sc = 1 / S.size;
  const pers = S.rough / 100;
  const wk = (S.warp / 100) * S.size * 0.9;
  const T = S.peakTile;

  // Peaks: at most one per tile, placed and sized by a hash of the tile.
  // Coordinates are (x, y * 2) in cells, which corrects for the cell aspect.
  const peakAt = (px: number, py: number) => {
    const tx = Math.floor(px / T);
    const ty = Math.floor(py / T);
    let h = 0;
    for (let j = ty - 1; j <= ty + 1; j++) {
      for (let i = tx - 1; i <= tx + 1; i++) {
        if (hash3(i, j, S.seed) >= S.peakChance) continue;
        const cx = (i + 0.15 + 0.7 * hash3(i, j, S.seed + 1)) * T;
        const cy = (j + 0.15 + 0.7 * hash3(i, j, S.seed + 2)) * T;
        const ph = (S.peakH / 100) * (0.6 + 0.4 * hash3(i, j, S.seed + 3));
        const pw = S.peakW * (0.7 + 0.6 * hash3(i, j, S.seed + 4));
        const dx = px - cx;
        const dy = py - cy;
        h += ph * Math.exp(-(dx * dx + dy * dy) / (2 * pw * pw));
      }
    }
    return h;
  };

  const sample = (wx: number, wy: number, out: Float32Array, i: number) => {
    const x = wx / 2;
    const y = wy / 4;
    let px = x;
    let py = y * 2;
    if (wk) {
      px += (nw1(x * sc * 0.6, py * sc * 0.6) - 0.5) * wk * 2;
      py += (nw2(x * sc * 0.6 + 9, py * sc * 0.6 + 9) - 0.5) * wk * 2;
    }
    let a = 0;
    let b = 0;
    let amp = 1;
    let fr = sc;
    let tot = 0;
    for (let o = 0; o < S.oct; o++) {
      const v = n(px * fr + o * 17.3, py * fr - o * 9.1);
      let rv = 1 - Math.abs(2 * v - 1);
      rv *= rv;
      a += v * amp;
      b += rv * amp;
      tot += amp;
      amp *= pers;
      fr *= 2;
    }
    const peak = peakAt(x, y * 2);
    out[i] = a / tot + peak;
    out[i + 1] = b / tot + peak;
  };

  // A fixed normalisation for the whole world, so contours don't rescale as
  // the view moves: robust min/max (0.2nd / 99.8th percentile) of a large random
  // sample, per ridge step.
  const r = rng(S.seed + 999);
  const N = 20000;
  const span = 6000; // cells
  const pts = new Float32Array(N * 2);
  for (let k = 0; k < N; k++)
    sample(r() * span * 2, r() * span * 4, pts, k * 2);
  let lo = Infinity;
  let hi = -Infinity;
  for (let k = 0; k < N * 2; k++) {
    lo = Math.min(lo, pts[k]);
    hi = Math.max(hi, pts[k]);
  }
  const norm: { min: number; max: number }[] = [];
  const blend = new Float32Array(N);
  for (let s = 0; s <= NORM_STEPS; s++) {
    const R = s / NORM_STEPS;
    for (let k = 0; k < N; k++)
      blend[k] = pts[k * 2] + (pts[k * 2 + 1] - pts[k * 2]) * R;
    blend.sort();
    norm.push({
      min: blend[Math.floor(N * 0.002)],
      max: blend[Math.floor(N * 0.998)],
    });
  }
  // Unsampled extremes can exceed the sample's range; leave headroom for packing.
  const pad = (hi - lo) * 0.25;
  return { settings: S, sample, lo: lo - pad, hi: hi + pad, norm };
}

export function normAt(world: TopoWorld, ridge: number) {
  const s = Math.max(0, Math.min(1, ridge)) * NORM_STEPS;
  const i = Math.min(NORM_STEPS - 1, Math.floor(s));
  const f = s - i;
  const a = world.norm[i];
  const b = world.norm[i + 1];
  return { min: a.min + (b.min - a.min) * f, max: a.max + (b.max - a.max) * f };
}

/** A rectangle of ring-buffer samples to upload: x, y, width, height. */
export type SampleRect = [number, number, number, number];

const mod = (v: number, n: number) => ((v % n) + n) % n;

/**
 * The samples for a cols x rows window of the world whose top-left cell is
 * (ox, oy), stored as a ring buffer: world sample (wx, wy) lives at
 * (wx mod width, wy mod height). Moving the window only samples the newly
 * exposed strips, and only those strips need uploading. Interleaved
 * [smooth, ridged] per sample.
 */
export class FieldWindow {
  readonly world: TopoWorld;
  readonly cols: number;
  readonly rows: number;
  readonly width: number; // cols * 2 + 1 samples
  readonly height: number; // rows * 4 + 1 samples
  ox: number;
  oy: number;
  readonly data: Float32Array;
  /** Ring-space rectangles changed since the last takeDirty() */
  private dirty: SampleRect[] = [];

  constructor(
    world: TopoWorld,
    cols: number,
    rows: number,
    ox: number,
    oy: number,
  ) {
    this.world = world;
    this.cols = cols;
    this.rows = rows;
    this.width = cols * 2 + 1;
    this.height = rows * 4 + 1;
    this.ox = ox;
    this.oy = oy;
    this.data = new Float32Array(this.width * this.height * 2);
    this.fillWorld(ox * 2, oy * 4, this.width, this.height);
  }

  /** Index into `data` of a sample, in window-relative sample coordinates. */
  index(fx: number, fy: number) {
    const x = mod(this.ox * 2 + fx, this.width);
    const y = mod(this.oy * 4 + fy, this.height);
    return (y * this.width + x) * 2;
  }

  // Sample a world-space rectangle into the ring, and mark it dirty.
  private fillWorld(wx0: number, wy0: number, w: number, h: number) {
    const { width: W, height: H, data, world } = this;
    for (let wy = wy0; wy < wy0 + h; wy++) {
      const row = mod(wy, H) * W;
      for (let wx = wx0; wx < wx0 + w; wx++)
        world.sample(wx, wy, data, (row + mod(wx, W)) * 2);
    }
    // Mark the ring-space rectangle(s); a span can wrap, giving up to 2 per axis
    const spans = (v0: number, len: number, n: number): [number, number][] => {
      if (len >= n) return [[0, n]];
      const s = mod(v0, n);
      return s + len <= n
        ? [[s, len]]
        : [
            [s, n - s],
            [0, len - (n - s)],
          ];
    };
    for (const [x, xw] of spans(wx0, w, W))
      for (const [y, yh] of spans(wy0, h, H)) this.dirty.push([x, y, xw, yh]);
  }

  /** Move the window; samples only what wasn't already covered. */
  moveTo(ox: number, oy: number) {
    if (ox === this.ox && oy === this.oy) return false;
    const { width: W, height: H } = this;
    const bx0 = this.ox * 2;
    const by0 = this.oy * 4;
    const bx = ox * 2;
    const by = oy * 4;
    this.ox = ox;
    this.oy = oy;
    if (Math.abs(bx - bx0) >= W || Math.abs(by - by0) >= H) {
      this.dirty = [];
      this.fillWorld(bx, by, W, H);
      return true;
    }
    // New rows (full width of the new window)
    if (by > by0) this.fillWorld(bx, by0 + H, W, by - by0);
    if (by < by0) this.fillWorld(bx, by, W, by0 - by);
    // New columns, for the rows not just filled
    const ry0 = Math.max(by, by0);
    const ry1 = Math.min(by + H, by0 + H);
    if (bx > bx0) this.fillWorld(bx0 + W, ry0, bx - bx0, ry1 - ry0);
    if (bx < bx0) this.fillWorld(bx, ry0, bx0 - bx, ry1 - ry0);
    return true;
  }

  /** Changed ring-space rectangles since the last call. */
  takeDirty() {
    const d = this.dirty;
    this.dirty = [];
    return d;
  }

  /** A ring-space rectangle packed as 16-bit values in RGBA8: rg = smooth, ba = ridged. */
  pack([x0, y0, w, h]: SampleRect) {
    const { data, world, width: W } = this;
    const out = new Uint8Array(w * h * 4);
    const span = world.hi - world.lo || 1;
    const q = (v: number) =>
      Math.round(Math.max(0, Math.min(1, (v - world.lo) / span)) * 65535);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = ((y0 + y) * W + x0 + x) * 2;
        const o = (y * w + x) * 4;
        const a = q(data[i]);
        const b = q(data[i + 1]);
        out[o] = a >> 8;
        out[o + 1] = a & 255;
        out[o + 2] = b >> 8;
        out[o + 3] = b & 255;
      }
    }
    return out;
  }
}

export type Summit = { x: number; y: number; text: string };

/** Highest local maxima in the window, labelled with their elevation, without overlapping. */
export function findSummits(
  field: FieldWindow,
  ridge: number,
  sea: number,
  count: number,
  spacingCols = field.cols,
): Summit[] {
  const S = field.world.settings;
  if (!count) return [];
  const { cols: W, rows: H, data } = field;
  const { min, max } = normAt(field.world, ridge);
  const span = max - min || 1;
  const G = (x: number, y: number) => {
    const i = field.index(x * 2, y * 4);
    const h = data[i] + (data[i + 1] - data[i]) * ridge;
    return (h - min) / span;
  };
  const elev = (h: number) => Math.round(((h - sea) / (1 - sea)) * S.maxE);

  const rx = Math.max(4, Math.round(spacingCols / 14));
  const ry = Math.max(2, Math.round(rx / 2));
  const cand: { x: number; y: number; h: number }[] = [];
  for (let y = 1; y < H; y++) {
    for (let x = 1; x < W; x++) {
      const h = G(x, y);
      if (h < sea + (1 - sea) * 0.35) continue;
      // Cheap 4-neighbour test first; most cells fail it.
      if (
        G(x - 1, y) > h ||
        G(x + 1, y) > h ||
        G(x, y - 1) > h ||
        G(x, y + 1) > h
      )
        continue;
      let ok = true;
      for (let dy = -ry; dy <= ry && ok; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy > H) continue;
        for (let dx = -rx; dx <= rx; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx > W || (!dx && !dy)) continue;
          if (G(xx, yy) > h) {
            ok = false;
            break;
          }
        }
      }
      if (ok) cand.push({ x, y, h });
    }
  }
  cand.sort((p, q) => q.h - p.h);

  const used = new Set<number>();
  const out: Summit[] = [];
  for (const p of cand) {
    if (out.length >= count) break;
    const text = "▲" + elev(p.h).toLocaleString("en").replace(/,/g, " ");
    const sx = Math.min(p.x, W - text.length);
    const yy = Math.min(p.y, H - 1);
    if (sx < 0) continue;
    let clash = false;
    for (let i = -1; i <= text.length; i++)
      for (let d = -1; d <= 1; d++)
        if (used.has((yy + d) * 1000 + sx + i)) clash = true;
    if (clash) continue;
    for (let i = 0; i < text.length; i++) used.add(yy * 1000 + sx + i);
    out.push({ x: sx, y: yy, text });
  }
  return out;
}
