// ASCII topographic terrain.
//
// The height at any point is a blend of two noise fields:
//   h = (1 - ridge) * smooth + ridge * ridged  (+ peaks, already folded into both)
// Because that blend is linear in `ridge`, both fields are built once here and the
// shader mixes them every frame, so ridges and sea level can animate for free.

export type TopoSettings = {
  seed: number;
  size: number; // feature size, in cells
  oct: number;
  rough: number;
  warp: number;
  island: number;
  peaks: number;
  peakH: number;
  peakW: number;
  labels: number;
  levels: number;
  index: number;
  maxE: number;
};

export const defaultTopoSettings = (cols: number): TopoSettings => ({
  seed: Math.floor(Math.random() * 99999),
  size: Math.max(24, Math.min(70, cols / 3.5)),
  oct: 4,
  rough: 50,
  warp: 25,
  island: 0,
  peaks: Math.max(3, Math.round(cols / 50)),
  peakH: 80,
  peakW: Math.max(9, cols / 12),
  labels: Math.max(3, Math.round(cols / 50)),
  levels: 12,
  index: 5,
  maxE: 3200,
});

/** Two height fields sampled at 2x4 points per text cell (one per braille dot). */
export type TopoField = {
  cols: number;
  rows: number;
  width: number; // cols * 2 + 1
  height: number; // rows * 4 + 1
  smooth: Float32Array;
  ridged: Float32Array;
  lo: number; // value range across both fields, for packing
  hi: number;
  norm: { min: number; max: number }[]; // min/max of the blend, per ridge step
};

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

export function buildField(W: number, H: number, S: TopoSettings): TopoField {
  const FW = W * 2 + 1;
  const FH = H * 4 + 1;
  const n = valueNoise(S.seed);
  const nw1 = valueNoise(S.seed + 101);
  const nw2 = valueNoise(S.seed + 202);
  const r = rng(S.seed + 7);
  const sc = 1 / S.size;
  const pers = S.rough / 100;
  const wk = (S.warp / 100) * S.size * 0.9;
  const I = S.island / 100;

  const pk: { x: number; y: number; h: number; w: number }[] = [];
  for (let i = 0; i < S.peaks; i++) {
    pk.push({
      x: W * (0.1 + 0.8 * r()),
      y: H * 2 * (0.15 + 0.7 * r()),
      h: (S.peakH / 100) * (0.6 + 0.4 * r()),
      w: S.peakW * (0.7 + 0.6 * r()),
    });
  }

  const smooth = new Float32Array(FW * FH);
  const ridged = new Float32Array(FW * FH);
  let lo = Infinity;
  let hi = -Infinity;
  for (let fy = 0; fy < FH; fy++) {
    for (let fx = 0; fx < FW; fx++) {
      const x = fx / 2;
      const y = fy / 4;
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
      // Terms that don't depend on the ridge blend go into both fields.
      let extra = 0;
      for (const p of pk) {
        const dx = x - p.x;
        const dy = y * 2 - p.y;
        extra += p.h * Math.exp(-(dx * dx + dy * dy) / (2 * p.w * p.w));
      }
      if (I) {
        const dx = (x - W / 2) / (W / 2);
        const dy = (y - H / 2) / (H / 2);
        extra -= I * 0.9 * Math.pow(Math.min(1.4, Math.hypot(dx, dy)), 2);
      }
      const i = fy * FW + fx;
      smooth[i] = a / tot + extra;
      ridged[i] = b / tot + extra;
      lo = Math.min(lo, smooth[i], ridged[i]);
      hi = Math.max(hi, smooth[i], ridged[i]);
    }
  }

  // The map is normalised to 0..1 for whatever ridge blend is showing; the
  // range isn't linear in the blend, so tabulate it.
  const norm: { min: number; max: number }[] = [];
  for (let s = 0; s <= NORM_STEPS; s++) {
    const R = s / NORM_STEPS;
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i < smooth.length; i++) {
      const h = smooth[i] + (ridged[i] - smooth[i]) * R;
      if (h < mn) mn = h;
      if (h > mx) mx = h;
    }
    norm.push({ min: mn, max: mx });
  }

  return {
    cols: W,
    rows: H,
    width: FW,
    height: FH,
    smooth,
    ridged,
    lo,
    hi,
    norm,
  };
}

export function normAt(field: TopoField, ridge: number) {
  const s = Math.max(0, Math.min(1, ridge)) * NORM_STEPS;
  const i = Math.min(NORM_STEPS - 1, Math.floor(s));
  const f = s - i;
  const a = field.norm[i];
  const b = field.norm[i + 1];
  return { min: a.min + (b.min - a.min) * f, max: a.max + (b.max - a.max) * f };
}

/** Both fields packed as 16-bit values into RGBA8: rg = smooth, ba = ridged. */
export function packField(field: TopoField) {
  const out = new Uint8Array(field.smooth.length * 4);
  const span = field.hi - field.lo || 1;
  for (let i = 0; i < field.smooth.length; i++) {
    const a = Math.round(((field.smooth[i] - field.lo) / span) * 65535);
    const b = Math.round(((field.ridged[i] - field.lo) / span) * 65535);
    out[i * 4] = a >> 8;
    out[i * 4 + 1] = a & 255;
    out[i * 4 + 2] = b >> 8;
    out[i * 4 + 3] = b & 255;
  }
  return out;
}

export type Summit = { x: number; y: number; text: string };

/** Highest local maxima, labelled with their elevation, without overlapping. */
export function findSummits(
  field: TopoField,
  S: TopoSettings,
  ridge: number,
  sea: number,
): Summit[] {
  if (!S.labels) return [];
  const { cols: W, rows: H, width: FW } = field;
  const { min, max } = normAt(field, ridge);
  const span = max - min || 1;
  const G = (x: number, y: number) => {
    const i = y * 4 * FW + x * 2;
    const h = field.smooth[i] + (field.ridged[i] - field.smooth[i]) * ridge;
    return (h - min) / span;
  };
  const elev = (h: number) => Math.round(((h - sea) / (1 - sea)) * S.maxE);

  const rx = Math.max(4, Math.round(W / 14));
  const ry = Math.max(2, Math.round(rx / 2));
  const cand: { x: number; y: number; h: number }[] = [];
  for (let y = 1; y < H; y++) {
    for (let x = 1; x < W; x++) {
      const h = G(x, y);
      if (h < sea + (1 - sea) * 0.35) continue;
      // Cheap 3x3 test first; most cells fail it.
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
    if (out.length >= S.labels) break;
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
