import React, { useEffect, useRef } from "react";
import { styled } from "../Stitches";
import {
  buildField,
  defaultTopoSettings,
  findSummits,
  normAt,
  packField,
  type Summit,
  type TopoField,
  type TopoSettings,
} from "./topo/terrain";

const Canvas = styled("canvas", {
  position: "fixed",
  top: 0,
  left: 0,
  width: "100vw",
  height: "100vh",
  zIndex: -1,
  pointerEvents: "none",
});

const vertexShaderSource = `
  attribute vec2 position;
  void main() {
    gl_Position = vec4(position, 0.0, 1.0);
  }
`;

// Pass 1: one fragment per text cell. Mixes the two height fields for the
// current ridge blend, finds which braille dots sit on a contour line (or
// whether the cell is under water) and writes the same per-cell encoding the
// display pass reads: r,a = atlas slot (lo, hi), g = kind, b = water depth.
const cellShaderSource = `
  precision highp float;

  uniform sampler2D uField;   // rg = smooth, ba = ridged, 16-bit each
  uniform vec2 uFieldSize;
  uniform sampler2D uLabels;  // summit labels: r,a = atlas slot, g = 1 where set
  uniform vec2 uGrid;
  uniform float uRidge;
  uniform float uSea;
  uniform vec2 uPackRange;    // lo, hi of the packed fields
  uniform vec2 uNorm;         // min, max of the blend at uRidge
  uniform float uLevels;
  uniform float uIndex;

  float height(vec2 f) {
      f = clamp(f, vec2(0.0), uFieldSize - 1.0);
      vec4 t = texture2D(uField, (f + 0.5) / uFieldSize) * 255.0;
      float a = (t.r * 256.0 + t.g) / 65535.0;
      float b = (t.b * 256.0 + t.a) / 65535.0;
      float h = uPackRange.x + mix(a, b, uRidge) * (uPackRange.y - uPackRange.x);
      return (h - uNorm.x) / (uNorm.y - uNorm.x);
  }

  float level(float h) {
      if (h < uSea) return -1.0;
      return min(uLevels - 1.0, floor((h - uSea) / (1.0 - uSea) * uLevels));
  }

  bool isIndex(float q) {
      return q > 0.0 && mod(q, uIndex) < 0.5;
  }

  vec4 encode(float slot, float kind, float depth) {
      return vec4(mod(slot, 256.0) / 255.0, kind / 255.0, depth, floor(slot / 256.0) / 255.0);
  }

  void main() {
      vec2 cell = floor(gl_FragCoord.xy);

      vec4 label = texture2D(uLabels, (cell + 0.5) / uGrid);
      if (label.g > 0.0) {
          gl_FragColor = encode(floor(label.r * 255.0 + 0.5) + 256.0 * floor(label.a * 255.0 + 0.5), 4.0, 0.0);
          return;
      }

      // Every sample this cell needs (8 dots, their neighbours, and the
      // corners for the water test) lies in a 4x6 block from (-1,-1) to (2,4)
      // relative to the cell's first dot. Read each sample once.
      vec2 origin = vec2(cell.x * 2.0, cell.y * 4.0);
      float h[24];
      float lv[24];
      for (int y = 0; y < 6; y++) {
          for (int x = 0; x < 4; x++) {
              float v = height(origin + vec2(float(x) - 1.0, float(y) - 1.0));
              h[y * 4 + x] = v;
              lv[y * 4 + x] = level(v);
          }
      }

      // Braille dot bits, in dot order (column-major, as in U+2800..28FF)
      float bits = 0.0;
      float best = -2.0;
      for (int j = 0; j < 4; j++) {
          for (int i = 0; i < 2; i++) {
              // Block index of this dot is (j + 1) * 4 + (i + 1). GLSL ES 1.0
              // only allows array indices built from loop indices and constants.
              float l = lv[(j + 1) * 4 + i + 1];
              if (l < 0.0) continue;
              // height() clamps, so at the left/top edge the neighbour is the
              // sample itself, as before
              bool edge =
                lv[(j + 1) * 4 + i + 2] < l || lv[(j + 2) * 4 + i + 1] < l ||
                lv[(j + 1) * 4 + i] < l || lv[j * 4 + i + 1] < l;
              if (!edge) continue;
              float bit = j == 3 ? (i == 0 ? 64.0 : 128.0) : pow(2.0, float(j + i * 3));
              bits += bit;
              if (best < 0.0 || (isIndex(l) && !isIndex(best)) || (isIndex(l) == isIndex(best) && l > best)) best = l;
          }
      }

      if (bits > 0.0) {
          float kind = best == 0.0 ? 1.0 : isIndex(best) ? 3.0 : 2.0;
          gl_FragColor = encode(bits + (kind == 3.0 ? 256.0 : 0.0), kind, 0.0);
          return;
      }

      // Corners (0,0), (2,0), (2,4), (0,4) of the cell, offset by the block origin
      float m = (h[5] + h[7] + h[23] + h[21]) / 4.0;
      if (m < uSea) {
          gl_FragColor = encode(0.0, 5.0, min(1.0, (uSea - m) / max(uSea, 0.01)));
          return;
      }
      gl_FragColor = encode(0.0, 0.0, 0.0);
  }
`;

const fragmentShaderSource = `
  precision highp float;

  uniform vec2 iResolution;
  uniform float iTime;

  #define bayerIntensity 0.7
  #define bayerScale 1.0
  #define errorDiffusionIntensity 0.02
  #define errorDiffusionScale 4.
  #define errorDiffusionIntensity2 0.05
  #define errorDiffusionScale2 1.
  #define colorNum 128.0  // Number of color levels per channel

  // Bayer matrix (4x4), built from the 2x2 one:
  // [0 8 2 10; 12 4 14 6; 3 11 1 9; 15 7 13 5] / 16
  float bayer2(vec2 p) {
      return mod(p.x * 2.0 + p.y * 3.0, 4.0);
  }
  float bayer4x4(vec2 position) {
      vec2 pos = floor(mod(position, 4.0));
      return (4.0 * bayer2(mod(pos, 2.0)) + bayer2(floor(pos / 2.0))) / 16.0;
  }

  uniform sampler2D uData;   // per-cell: r,a = glyph index (lo,hi), g = kind, b = water depth
  uniform sampler2D uAtlas;  // glyph atlas, coverage in alpha
  uniform vec2 uGrid;        // cols, rows
  uniform vec2 uCell;        // cell size in device px
  uniform vec2 uAtlasGrid;   // atlas slots across, down
  uniform vec3 uWaterGlyphs; // atlas indices of ~, ≈, ·
  uniform vec3 uContour;     // contour ink, drifting with the slow palette cycle

  #define topoStrength 1.0
  #define waterDensity 0.6
  #define shimmer 0.7

  float hash12(vec2 p) {
      vec3 p3 = fract(vec3(p.xyx) * .1031);
      p3 += dot(p3, p3.yzx + 33.33);
      return fract((p3.x + p3.y) * p3.z);
  }

  float glyphCoverage(float g, vec2 local) {
      vec2 slot = vec2(mod(g, uAtlasGrid.x), floor(g / uAtlasGrid.x));
      return texture2D(uAtlas, (slot + local) / uAtlasGrid).a;
  }

  // Error-diffusion-style threshold: one hash per (scale x scale) block.
  // (An earlier version also summed neighbour error, but its weights never
  // applied, so the error was always zero; this gives identical output.)
  float errorDiffusion(vec2 coord, float value, float scale) {
      vec2 noiseCoord = floor(coord / scale);
      float threshold = fract(sin(dot(noiseCoord, vec2(12.9898, 78.233))) * 43758.5453);
      return (value > threshold) ? 1.0 : 0.0;
  }

  void main() {
      vec2 fragCoord = gl_FragCoord.xy;
      vec2 uv = fragCoord / iResolution.xy;

      // Paper: the same olive gradient as the page behind it
      vec3 bgTop = vec3(33, 36, 10) / vec3(255);
      vec3 bgBottom = vec3(45, 48, 21) / vec3(255);
      vec3 color = mix(bgBottom, bgTop, uv.y);

      // Which text cell this pixel is in (row 0 at the top)
      vec2 p = vec2(fragCoord.x, iResolution.y - fragCoord.y);
      vec2 cell = floor(p / uCell);
      vec2 local = (p - cell * uCell) / uCell;

      if (cell.x < uGrid.x && cell.y < uGrid.y) {
          vec4 d = texture2D(uData, (cell + 0.5) / uGrid);
          float glyph = floor(d.r * 255.0 + 0.5) + 256.0 * floor(d.a * 255.0 + 0.5);
          float kind = floor(d.g * 255.0 + 0.5);
          float depth = d.b;

          vec3 cream = vec3(255, 180, 100) / vec3(255);
          vec3 shore = vec3(120, 150, 130) / vec3(255);
          vec3 water = vec3(95, 135, 125) / vec3(255);
          vec3 glint = vec3(185, 210, 175) / vec3(255);

          vec3 contour = uContour;

          vec3 ink = vec3(0.0);
          float alpha = 0.0;

          if (kind == 5.0) {
              float h1 = hash12(cell);
              float h2 = hash12(cell.yx + vec2(31.0, 17.0));
              float tt = iTime;
              float v = 0.0;
              if (h1 < waterDensity) {
                  float x = cell.x, y = cell.y;
                  float crest = (sin(x * .95 + y * 1.7 - tt * 1.6 + sin(x * .11 + y * .07 + tt * .35) * 2.2)
                      + .35 * sin(x * .23 - y * .4 + tt * .9)) / 1.35;
                  v = smoothstep(-.2, .9, crest);
                  glyph = depth > .45 ? uWaterGlyphs.y : uWaterGlyphs.x;
              } else if (h2 < .14 * shimmer) {
                  float ph = tt * (0.3 + h2 * 3.0) + h2 * 97.0;
                  float k = floor(ph);
                  float on = step(0.6, hash12(cell + k)) * (.55 + .45 * hash12(cell + k + 3.0));
                  v = on * sin(fract(ph) * 3.14159);
                  glyph = uWaterGlyphs.z;
              }
              float maxA = .42 + .28 * shimmer;
              alpha = min(.9, 1.3 * v * maxA * (1.0 - depth * .25));
              ink = mix(water, glint, min(.55, v * v * .6 * shimmer));
              if (depth < .12) ink = (shore + ink) * 0.5;
          } else if (kind == 1.0) {
              ink = shore; alpha = 0.8;
          } else if (kind == 2.0) {
              ink = contour; alpha = 0.8;
          } else if (kind == 3.0) {
              ink = cream; alpha = 1.0;
          } else if (kind == 4.0) {
              ink = vec3(252, 243, 230) / vec3(255); alpha = 1.0;
          }

          if (alpha > 0.0 && glyph > 0.0) {
              float cov = glyphCoverage(glyph, local);
              color = mix(color, ink, cov * alpha * topoStrength);
          }
      }

      // Calculate brightness
      float brightness = dot(color, vec3(0.299, 0.587, 0.114));

      // Apply error diffusion layer 1 (coarse)
      float dithered1 = errorDiffusion(fragCoord, brightness, errorDiffusionScale);
      float ditherDelta1 = (dithered1 - brightness) * errorDiffusionIntensity;
      color = color + vec3(ditherDelta1);

      // Recalculate brightness after first dither
      brightness = dot(color, vec3(0.299, 0.587, 0.114));

      // Apply error diffusion layer 2 (fine)
      float dithered2 = errorDiffusion(fragCoord, brightness, errorDiffusionScale2);
      float ditherDelta2 = (dithered2 - brightness) * errorDiffusionIntensity2;
      color = color + vec3(ditherDelta2);

      // Apply Bayer dithering to highlights only
      vec2 bayerCoord = fragCoord / bayerScale;
      float bayerValue = bayer4x4(bayerCoord) - 0.5;

      // Recalculate brightness for highlight mask
      brightness = dot(color, vec3(0.299, 0.587, 0.114));
      float highlightMask = smoothstep(0.3, 0.8, brightness);

      // Apply bayer threshold only to highlights
      color = color + bayerValue * bayerIntensity * highlightMask;

      // Quantize to discrete color levels (color banding)
      color.r = floor(color.r * (colorNum - 1.0) + 0.5) / (colorNum - 1.0);
      color.g = floor(color.g * (colorNum - 1.0) + 0.5) / (colorNum - 1.0);
      color.b = floor(color.b * (colorNum - 1.0) + 0.5) / (colorNum - 1.0);

      gl_FragColor = vec4(color, 1.0);
  }
`;

const FONT_PX = 11;
const LINE_HEIGHT = 1.22;
const ATLAS_COLS = 32;
const FONT_STACK = '"Iosevka SS05", Menlo, "DejaVu Sans Mono", monospace';
const LABEL_CHARS = "▲0123456789";
const WATER_GLYPHS = ["~", "≈", "·"];
const LABEL_INTERVAL_MS = 500;

// Slow warm drift on the contour ink, a nod to the old palette cycle:
// mixes from (200,120,70) toward rust (220,70,60).
const contourAt = (t: number) => {
  const cycle = Math.sin(t * 0.25);
  const k =
    ((Math.sign(cycle) * Math.pow(Math.abs(cycle), 0.6) + 1) / 2) * 0.35;
  return [200 + 20 * k, 120 - 50 * k, 70 - 10 * k].map((c) => c / 255);
};

// Ridges and sea level drift on slow, incommensurate sine waves.
const TAU = Math.PI * 2;
const ridgeAt = (t: number) =>
  0.35 + 0.25 * Math.sin((t * TAU) / 97) + 0.08 * Math.sin((t * TAU) / 41 + 2);
const seaAt = (t: number) =>
  0.18 + 0.07 * Math.sin((t * TAU) / 73 + 1) + 0.03 * Math.sin((t * TAU) / 29);

// Atlas layout: 0-255 braille, 256-511 bold braille (index contours), then
// bold label characters, then water glyphs.
type Atlas = {
  canvas: HTMLCanvasElement;
  rows: number;
  slot: Map<string, number>;
};

function buildAtlas(cellW: number, cellH: number, fontPx: number): Atlas {
  const glyphs: [string, number][] = [];
  for (let b = 0; b < 256; b++)
    glyphs.push([String.fromCharCode(0x2800 + b), 400]);
  for (let b = 0; b < 256; b++)
    glyphs.push([String.fromCharCode(0x2800 + b), 700]);
  for (const ch of LABEL_CHARS) glyphs.push([ch, 700]);
  for (const ch of WATER_GLYPHS) glyphs.push([ch, 400]);

  const slot = new Map<string, number>();
  const rows = Math.ceil(glyphs.length / ATLAS_COLS);
  const canvas = document.createElement("canvas");
  canvas.width = ATLAS_COLS * cellW;
  canvas.height = rows * cellH;
  const ctx = canvas.getContext("2d")!;
  ctx.fillStyle = "#fff";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  glyphs.forEach(([ch, weight], i) => {
    if (i >= 512) slot.set(ch, i);
    const x = (i % ATLAS_COLS) * cellW;
    const y = Math.floor(i / ATLAS_COLS) * cellH;
    ctx.save();
    ctx.beginPath();
    ctx.rect(x, y, cellW, cellH);
    ctx.clip();
    ctx.font = `${weight} ${fontPx}px ${FONT_STACK}`;
    ctx.fillText(ch, x + cellW / 2, y + cellH / 2);
    ctx.restore();
  });
  return { canvas, rows, slot };
}

function encodeLabels(
  cols: number,
  rows: number,
  summits: Summit[],
  atlas: Atlas,
) {
  const data = new Uint8Array(cols * rows * 4);
  for (const s of summits) {
    for (let k = 0; k < s.text.length; k++) {
      const i = (s.y * cols + s.x + k) * 4;
      const g = atlas.slot.get(s.text[k]) ?? 0; // spaces stay blank
      data[i] = g & 255;
      data[i + 1] = 1;
      data[i + 3] = g >> 8;
    }
  }
  return data;
}

function createTexture(gl: WebGLRenderingContext) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

function compile(gl: WebGLRenderingContext, type: number, src: string) {
  const sh = gl.createShader(type);
  if (!sh) return null;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    console.error("Shader compilation error:", gl.getShaderInfoLog(sh));
    return null;
  }
  return sh;
}

function link(gl: WebGLRenderingContext, vs: WebGLShader, fsSource: string) {
  const fs = compile(gl, gl.FRAGMENT_SHADER, fsSource);
  const program = gl.createProgram();
  if (!fs || !program) return null;
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error("Program linking error:", gl.getProgramInfoLog(program));
    return null;
  }
  return program;
}

export const ShaderBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const gl = canvas.getContext("webgl");
    if (!gl) {
      console.error("WebGL not supported");
      return;
    }

    const vs = compile(gl, gl.VERTEX_SHADER, vertexShaderSource);
    if (!vs) return;
    const cellProgram = link(gl, vs, cellShaderSource);
    const displayProgram = link(gl, vs, fragmentShaderSource);
    if (!cellProgram || !displayProgram) return;

    // Full-screen quad, shared by both passes (both use attribute location of "position")
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]),
      gl.STATIC_DRAW,
    );
    for (const program of [cellProgram, displayProgram]) {
      const loc = gl.getAttribLocation(program, "position");
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    }

    const cu = (name: string) => gl.getUniformLocation(cellProgram, name);
    const du = (name: string) => gl.getUniformLocation(displayProgram, name);

    const fieldTex = createTexture(gl);
    const labelTex = createTexture(gl);
    const cellTex = createTexture(gl);
    const atlasTex = createTexture(gl);
    const cellFbo = gl.createFramebuffer();

    gl.useProgram(cellProgram);
    gl.uniform1i(cu("uField"), 0);
    gl.uniform1i(cu("uLabels"), 1);
    gl.useProgram(displayProgram);
    gl.uniform1i(du("uData"), 0);
    gl.uniform1i(du("uAtlas"), 1);

    const reducedMotion = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    const seed = Math.floor(Math.random() * 99999);
    const startTime = performance.now();
    const now = () =>
      reducedMotion ? 0 : (performance.now() - startTime) / 1000;

    let field: TopoField | null = null;
    let settings: TopoSettings | null = null;
    let atlas: Atlas | null = null;
    let lastLabels = -Infinity;

    const uploadLabels = (t: number) => {
      if (!field || !settings || !atlas) return;
      const summits = findSummits(field, settings, ridgeAt(t), seaAt(t));
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, labelTex);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        field.cols,
        field.rows,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        encodeLabels(field.cols, field.rows, summits, atlas),
      );
    };

    const draw = () => {
      if (!field || !settings || !atlas) return;
      const t = now();
      const ridge = ridgeAt(t);
      const sea = seaAt(t);
      const norm = normAt(field, ridge);

      if (performance.now() - lastLabels > LABEL_INTERVAL_MS) {
        lastLabels = performance.now();
        uploadLabels(t);
      }

      // Pass 1: terrain -> per-cell glyphs
      gl.bindFramebuffer(gl.FRAMEBUFFER, cellFbo);
      gl.viewport(0, 0, field.cols, field.rows);
      gl.useProgram(cellProgram);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, fieldTex);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, labelTex);
      gl.uniform1f(cu("uRidge"), ridge);
      gl.uniform1f(cu("uSea"), sea);
      gl.uniform2f(cu("uNorm"), norm.min, norm.max);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      // Pass 2: glyphs -> screen, with the dither stack
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.useProgram(displayProgram);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, cellTex);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, atlasTex);
      gl.uniform2f(du("iResolution"), canvas.width, canvas.height);
      gl.uniform1f(du("iTime"), t);
      gl.uniform3fv(du("uContour"), contourAt(t));
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    };

    // (Re)build the terrain and glyph atlas for the current viewport.
    const rebuild = () => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(window.innerWidth * dpr);
      canvas.height = Math.round(window.innerHeight * dpr);

      const fontPx = FONT_PX * dpr;
      const measure = document.createElement("canvas").getContext("2d")!;
      measure.font = `400 ${fontPx}px ${FONT_STACK}`;
      const cellW = Math.max(
        1,
        Math.round(measure.measureText("MMMMMMMMMM").width / 10),
      );
      const cellH = Math.max(1, Math.round(fontPx * LINE_HEIGHT));
      const cols = Math.ceil(canvas.width / cellW);
      const rows = Math.ceil(canvas.height / cellH);

      settings = { ...defaultTopoSettings(cols), seed };
      field = buildField(cols, rows, settings);
      atlas = buildAtlas(cellW, cellH, fontPx);

      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, fieldTex);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        field.width,
        field.height,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        packField(field),
      );
      gl.bindTexture(gl.TEXTURE_2D, atlasTex);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        atlas.canvas,
      );
      gl.bindTexture(gl.TEXTURE_2D, cellTex);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        cols,
        rows,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        null,
      );
      gl.bindFramebuffer(gl.FRAMEBUFFER, cellFbo);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        cellTex,
        0,
      );
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      gl.useProgram(cellProgram);
      gl.uniform2f(cu("uFieldSize"), field.width, field.height);
      gl.uniform2f(cu("uGrid"), cols, rows);
      gl.uniform2f(cu("uPackRange"), field.lo, field.hi);
      gl.uniform1f(cu("uLevels"), settings.levels);
      gl.uniform1f(cu("uIndex"), settings.index);

      gl.useProgram(displayProgram);
      gl.uniform2f(du("uGrid"), cols, rows);
      gl.uniform2f(du("uCell"), cellW, cellH);
      gl.uniform2f(du("uAtlasGrid"), ATLAS_COLS, atlas.rows);
      const [tilde, approx, dot] = WATER_GLYPHS.map(
        (c) => atlas!.slot.get(c) ?? 0,
      );
      gl.uniform3f(du("uWaterGlyphs"), tilde, approx, dot);

      lastLabels = -Infinity;
      draw();
    };

    let resizeTimer: ReturnType<typeof setTimeout> | undefined;
    const handleResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(rebuild, 150);
    };

    rebuild();
    window.addEventListener("resize", handleResize);
    document.fonts?.ready.then(rebuild);

    // Animation loop at 30fps
    let raf = 0;
    let lastFrameTime = 0;
    const frameInterval = 1000 / 30;
    const render = (frameTime: number) => {
      raf = requestAnimationFrame(render);
      const elapsed = frameTime - lastFrameTime;
      if (elapsed < frameInterval) return;
      lastFrameTime = frameTime - (elapsed % frameInterval);
      draw();
    };
    if (!reducedMotion) raf = requestAnimationFrame(render);

    return () => {
      window.removeEventListener("resize", handleResize);
      clearTimeout(resizeTimer);
      cancelAnimationFrame(raf);
      for (const tex of [fieldTex, labelTex, cellTex, atlasTex])
        gl.deleteTexture(tex);
      gl.deleteFramebuffer(cellFbo);
      gl.deleteProgram(cellProgram);
      gl.deleteProgram(displayProgram);
    };
  }, []);

  return <Canvas ref={canvasRef} />;
};
