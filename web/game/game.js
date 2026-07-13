/* Antiphon playable street demo.
 *
 * Audio model: one looped low-band noise stream x feeds two convolver
 * chains per position — h_off (ANC off) and h_on (speaker anti-noise
 * folded in at bake time). ANC toggling crossfades the two chains; moving
 * crossfades convolvers between grid cells. The pass-through highs loop
 * gets distance attenuation. All IRs come from FDTD wave simulation.
 */
'use strict';

const $ = (id) => document.getElementById(id);

// ---------- float16 decode ----------
function f16ToF32Array(u16) {
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = (h & 0x8000) ? -1 : 1;
    const e = (h >> 10) & 0x1f;
    const m = h & 0x3ff;
    if (e === 0) out[i] = s * m * 2 ** -24;
    else if (e === 31) out[i] = m ? NaN : s * Infinity;
    else out[i] = s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

// ---------- state ----------
const G = {
  meta: null, irsRaw: null, noiseBuf: null, highsBuf: null,
  imgOff: new Image(), imgOn: new Image(),
  ctx: null, master: null, analyser: null,
  lowGain: null, highGain: null, gOff: null, gOn: null,
  slots: { off: [], on: [] },
  cellCache: new Map(),
  anc: false, fieldAlpha: 0, canopy: false, canopyAlpha: 0,
  px: 4.0, py: 1.0, trail: [],
  keys: {}, cell: -1, started: false,
};

// ---------- asset loading ----------
async function loadAssets() {
  const meta = await (await fetch('assets/meta.json')).json();
  G.meta = meta;
  const irs = await (await fetch('assets/irs.bin')).arrayBuffer();
  G.irsRaw = new Uint16Array(irs);
  const noise = await (await fetch('assets/noise_low.bin')).arrayBuffer();
  G.noiseF32 = f16ToF32Array(new Uint16Array(noise));
  G.highsArr = await (await fetch('assets/highs.ogg')).arrayBuffer();
  G.imgOff.src = 'assets/field_off.png';
  G.imgOn.src = 'assets/field_on.png';
  $('credit').textContent = meta.credit;
}

function cellIndex(x, y) {
  const gx = G.meta.grid.x, gy = G.meta.grid.y;
  let ix = Math.round((x - gx[0]) / (gx[1] - gx[0]));
  let iy = Math.round((y - gy[0]) / (gy[1] - gy[0]));
  ix = Math.max(0, Math.min(gx.length - 1, ix));
  iy = Math.max(0, Math.min(gy.length - 1, iy));
  return ix * gy.length + iy;
}

function cellBuffers(idx) {
  if (G.cellCache.has(idx)) return G.cellCache.get(idx);
  const L = G.meta.ir_len;
  const base = idx * 2 * L;
  const mk = (off) => {
    const f32 = f16ToF32Array(G.irsRaw.subarray(base + off * L,
                                                base + (off + 1) * L));
    const buf = G.ctx.createBuffer(1, L, G.meta.fs);
    buf.copyToChannel(f32, 0);
    return buf;
  };
  const pair = { off: mk(0), on: mk(1) };
  G.cellCache.set(idx, pair);
  if (G.cellCache.size > 200) {
    const first = G.cellCache.keys().next().value;
    if (first !== idx) G.cellCache.delete(first);
  }
  return pair;
}

// ---------- audio graph ----------
function showError(msg) {
  const card = document.getElementById('startcard');
  card.innerHTML = '<h2>Audio error</h2><p style="color:#f85149">' + msg +
    '</p><p>Open the devtools console for details.</p>';
  document.getElementById('overlay').classList.remove('hidden');
}

function startAudio() {
  // Context at the asset rate: convolver buffers match natively
  let ctx;
  try {
    ctx = new (window.AudioContext || window.webkitAudioContext)(
      { sampleRate: G.meta.fs });
  } catch (e) {
    ctx = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (ctx.state === 'suspended') ctx.resume();
  G.ctx = ctx;

  const noiseBuf = ctx.createBuffer(1, G.noiseF32.length, G.meta.fs);
  noiseBuf.copyToChannel(G.noiseF32, 0);
  G.noiseSrc = ctx.createBufferSource();
  G.noiseSrc.buffer = noiseBuf;
  G.noiseSrc.loop = true;

  G.gOff = ctx.createGain();
  G.gOn = ctx.createGain();
  G.gOff.gain.value = 1;
  G.gOn.gain.value = 0;

  G.lowGain = ctx.createGain();
  G.lowGain.gain.value = G.meta.low_calibration * G.meta.low_boost;
  G.gOff.connect(G.lowGain);
  G.gOn.connect(G.lowGain);

  G.highGain = ctx.createGain();
  G.highGain.gain.value = 0;

  // Passive absorptive canopy: attenuation rising with frequency
  // (~3 dB @ 500 Hz to ~15 dB @ 4 kHz, typical absorptive screen)
  G.gHDry = ctx.createGain();
  G.gHCan = ctx.createGain();
  G.gHDry.gain.value = 1;
  G.gHCan.gain.value = 0;
  const shelf1 = ctx.createBiquadFilter();
  shelf1.type = 'highshelf';
  shelf1.frequency.value = 600;
  shelf1.gain.value = -8;
  const shelf2 = ctx.createBiquadFilter();
  shelf2.type = 'highshelf';
  shelf2.frequency.value = 1800;
  shelf2.gain.value = -7;
  G.highGain.connect(G.gHDry);
  G.highGain.connect(shelf1);
  shelf1.connect(shelf2);
  shelf2.connect(G.gHCan);

  G.master = ctx.createGain();
  G.master.gain.value = 0.4 * ($('vol').value / 60);
  G.analyser = ctx.createAnalyser();
  G.analyser.fftSize = 2048;
  G.lowGain.connect(G.master);
  G.gHDry.connect(G.master);
  G.gHCan.connect(G.master);
  G.master.connect(G.analyser);
  G.analyser.connect(ctx.destination);

  ctx.decodeAudioData(G.highsArr.slice(0)).catch((e) => {
    console.error('highs decode failed', e);
    return null;
  }).then((buf) => {
    if (!buf) return;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.loop = true;
    src.connect(G.highGain);
    src.start();
  });
  console.log('audio started: ctx rate', ctx.sampleRate);

  G.noiseSrc.start();
  switchCell(cellIndex(G.px, G.py), true);
  updateHighsGain();
}

function makeSlot(buffer, dest, gainVal) {
  const conv = G.ctx.createConvolver();
  conv.normalize = false;
  conv.buffer = buffer;
  const g = G.ctx.createGain();
  g.gain.value = gainVal;
  G.noiseSrc.connect(conv);
  conv.connect(g);
  g.connect(dest);
  return { conv, g };
}

function switchCell(idx, instant) {
  if (idx === G.cell) return;
  G.cell = idx;
  const bufs = cellBuffers(idx);
  const t = G.ctx.currentTime;
  const FADE = instant ? 0.02 : 0.12;
  for (const state of ['off', 'on']) {
    const dest = state === 'off' ? G.gOff : G.gOn;
    for (const s of G.slots[state]) {
      s.g.gain.cancelScheduledValues(t);
      s.g.gain.setValueAtTime(s.g.gain.value, t);
      s.g.gain.linearRampToValueAtTime(0, t + FADE);
      setTimeout(() => { try { s.conv.disconnect(); s.g.disconnect(); } catch (e) {} },
                 FADE * 1000 + 150);
    }
    const slot = makeSlot(bufs[state], dest, 0);
    slot.g.gain.setValueAtTime(0, t);
    slot.g.gain.linearRampToValueAtTime(1, t + FADE);
    G.slots[state] = [slot];
  }
}

function updateHighsGain() {
  const [nx, ny] = G.meta.noise_pos;
  const r = Math.hypot(G.px - nx, G.py - ny);
  const r0 = Math.hypot(4.0 - nx, 1.0 - ny);
  const g = Math.sqrt(r0 / Math.max(r, 0.5));
  G.highGain.gain.setTargetAtTime(
    G.meta.highs_scale * g * 0.9, G.ctx.currentTime, 0.1);
}

function setAnc(on) {
  G.anc = on;
  const t = G.ctx.currentTime;
  G.gOn.gain.cancelScheduledValues(t);
  G.gOff.gain.cancelScheduledValues(t);
  G.gOn.gain.setValueAtTime(G.gOn.gain.value, t);
  G.gOff.gain.setValueAtTime(G.gOff.gain.value, t);
  G.gOn.gain.linearRampToValueAtTime(on ? 1 : 0, t + 0.5);
  G.gOff.gain.linearRampToValueAtTime(on ? 0 : 1, t + 0.5);
  const btn = $('ancbtn');
  btn.className = on ? 'on' : 'off';
  btn.innerHTML = on ? 'ANC&nbsp;ON' : 'ANC&nbsp;OFF';
  $('meterbar').className = on ? 'on' : '';
}

function setCanopy(on) {
  G.canopy = on;
  const t = G.ctx.currentTime;
  for (const [g, v] of [[G.gHDry, on ? 0 : 1], [G.gHCan, on ? 1 : 0]]) {
    g.gain.cancelScheduledValues(t);
    g.gain.setValueAtTime(g.gain.value, t);
    g.gain.linearRampToValueAtTime(v, t + 0.4);
  }
  const btn = $('canbtn');
  btn.className = on ? 'on' : 'off';
  btn.innerHTML = on ? 'CANOPY&nbsp;ON' : 'CANOPY&nbsp;OFF';
}

// ---------- world / rendering ----------
const SPEED = 2.2; // m/s

function step(dt) {
  let vx = 0, vy = 0;
  if (G.keys.w || G.keys.ArrowUp) vy += 1;
  if (G.keys.s || G.keys.ArrowDown) vy -= 1;
  if (G.keys.a || G.keys.ArrowLeft) vx -= 1;
  if (G.keys.d || G.keys.ArrowRight) vx += 1;
  if (vx || vy) {
    const n = Math.hypot(vx, vy);
    G.px += (vx / n) * SPEED * dt;
    G.py += (vy / n) * SPEED * dt;
    const [x0, x1, y0, y1] = G.meta.walk_bounds;
    const hw = G.meta.street_halfwidth - 0.35;
    G.px = Math.max(x0, Math.min(x1, G.px));
    G.py = Math.max(Math.max(y0, -hw), Math.min(Math.min(y1, hw), G.py));
    if (!G.trail.length ||
        Math.hypot(G.px - G.trail[G.trail.length - 1][0],
                   G.py - G.trail[G.trail.length - 1][1]) > 0.3) {
      G.trail.push([G.px, G.py]);
      if (G.trail.length > 220) G.trail.shift();
    }
    if (G.started) {
      switchCell(cellIndex(G.px, G.py), false);
      updateHighsGain();
    }
  }
  G.fieldAlpha += ((G.anc ? 1 : 0) - G.fieldAlpha) * Math.min(1, dt * 3);
  G.canopyAlpha += ((G.canopy ? 1 : 0) - G.canopyAlpha) * Math.min(1, dt * 3);
}

function world2px(cv, x, y) {
  const [ex0, ex1, ey0, ey1] = G.meta.field_extent;
  return [((x - ex0) / (ex1 - ex0)) * cv.width,
          (1 - (y - ey0) / (ey1 - ey0)) * cv.height];
}

function draw() {
  const cv = $('view');
  const c = cv.getContext('2d');
  c.clearRect(0, 0, cv.width, cv.height);
  if (G.imgOff.complete) {
    c.globalAlpha = 1;
    c.drawImage(G.imgOff, 0, 0, cv.width, cv.height);
    if (G.imgOn.complete && G.fieldAlpha > 0.01) {
      c.globalAlpha = G.fieldAlpha;
      c.drawImage(G.imgOn, 0, 0, cv.width, cv.height);
      c.globalAlpha = 1;
    }
  }

  const m = G.meta;
  // noise source
  let [sx, sy] = world2px(cv, m.noise_pos[0], m.noise_pos[1]);
  c.fillStyle = '#ff5d5d';
  c.strokeStyle = '#fff';
  star(c, sx, sy, 11);
  label(c, 'traffic noise', sx - 36, sy - 16);

  // speakers
  for (const [x, y] of m.speakers) {
    const [px, py] = world2px(cv, x, y);
    c.fillStyle = G.fieldAlpha > 0.5 ? '#3ddbd9' : '#444c56';
    tri(c, px, py, 8);
  }
  // mics + bench
  c.fillStyle = '#a5d6ff';
  for (const [x, y] of m.mics) {
    const [px, py] = world2px(cv, x, y);
    c.beginPath(); c.arc(px, py, 2, 0, 7); c.fill();
  }
  const [bx, by] = world2px(cv, m.bench[0], m.bench[1]);
  c.fillStyle = '#a5d6ff';
  c.fillRect(bx - 6, by - 6, 12, 12);
  label(c, 'quiet-zone bench', bx - 55, by + 24);

  // absorptive canopy strips along both facades
  if (G.canopyAlpha > 0.02) {
    const hw = m.street_halfwidth;
    for (const sgn of [1, -1]) {
      const [ax, ay] = world2px(cv, 1.6, sgn * hw);
      const [bx2, by2] = world2px(cv, m.domain[0] - 1.6, sgn * (hw - 1.1));
      c.fillStyle = `rgba(87, 171, 90, ${0.30 * G.canopyAlpha})`;
      c.fillRect(Math.min(ax, bx2), Math.min(ay, by2),
                 Math.abs(bx2 - ax), Math.abs(by2 - ay));
    }
    c.globalAlpha = G.canopyAlpha;
    label(c, 'absorptive canopy (passive)', cv.width / 2 - 100, 36);
    c.globalAlpha = 1;
  }

  // trail + walker
  c.strokeStyle = 'rgba(230,237,243,0.35)';
  c.lineWidth = 1.5;
  c.beginPath();
  G.trail.forEach(([x, y], i) => {
    const [px, py] = world2px(cv, x, y);
    i ? c.lineTo(px, py) : c.moveTo(px, py);
  });
  c.stroke();
  const [wx, wy] = world2px(cv, G.px, G.py);
  c.fillStyle = '#fff';
  c.strokeStyle = '#000';
  c.beginPath(); c.arc(wx, wy, 7, 0, 7); c.fill(); c.stroke();
  label(c, 'you', wx - 12, wy - 14);
}

function star(c, x, y, r) {
  c.beginPath();
  for (let i = 0; i < 10; i++) {
    const rr = i % 2 ? r / 2.4 : r;
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    i ? c.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a))
      : c.moveTo(x + rr * Math.cos(a), y + rr * Math.sin(a));
  }
  c.closePath(); c.fill(); c.stroke();
}
function tri(c, x, y, r) {
  c.beginPath();
  c.moveTo(x, y - r); c.lineTo(x - r, y + r); c.lineTo(x + r, y + r);
  c.closePath(); c.fill();
}
function label(c, txt, x, y) {
  c.font = 'bold 13px ui-monospace, monospace';
  c.lineWidth = 3;
  c.strokeStyle = '#0d1117';
  c.strokeText(txt, x, y);
  c.fillStyle = '#e6edf3';
  c.fillText(txt, x, y);
}

// ---------- meter ----------
const meterBuf = new Float32Array(2048);
let meterDb = -60;
function meter() {
  if (!G.analyser) return;
  G.analyser.getFloatTimeDomainData(meterBuf);
  let s = 0;
  for (let i = 0; i < meterBuf.length; i++) s += meterBuf[i] * meterBuf[i];
  const db = 20 * Math.log10(Math.sqrt(s / meterBuf.length) + 1e-9);
  meterDb += (db - meterDb) * 0.25;
  const frac = Math.max(0, Math.min(1, (meterDb + 55) / 45));
  $('meterbar').style.width = `${frac * 100}%`;
  $('meterdb').textContent = `${meterDb.toFixed(0)} dB`;
}

// ---------- main ----------
let last = 0;
function loop(t) {
  const dt = Math.min(0.05, (t - last) / 1000 || 0.016);
  last = t;
  if (G.meta) {
    step(dt);
    draw();
    meter();
  }
  requestAnimationFrame(loop);
}

window.addEventListener('keydown', (e) => {
  if (e.key === ' ') {
    e.preventDefault();
    if (G.started) setAnc(!G.anc);
    return;
  }
  if (e.key === 'c' || e.key === 'C') {
    if (G.started) setCanopy(!G.canopy);
    return;
  }
  G.keys[e.key.toLowerCase()] = true;
  G.keys[e.key] = true;
});
window.addEventListener('keyup', (e) => {
  G.keys[e.key.toLowerCase()] = false;
  G.keys[e.key] = false;
});

$('ancbtn').addEventListener('click', () => G.started && setAnc(!G.anc));
$('canbtn').addEventListener('click', () => G.started && setCanopy(!G.canopy));
$('vol').addEventListener('input', () => {
  if (G.master) G.master.gain.value = 0.4 * ($('vol').value / 60);
});
$('startbtn').addEventListener('click', () => {
  if (!G.loaded) return;
  try {
    startAudio();
    G.started = true;
    $('overlay').classList.add('hidden');
  } catch (e) {
    console.error(e);
    showError(String(e));
  }
});

$('startbtn').disabled = true;
$('startbtn').textContent = 'loading assets\u2026';
loadAssets().then(() => {
  G.loaded = true;
  $('startbtn').disabled = false;
  $('startbtn').textContent = 'Enter the street';
}).catch((e) => {
  console.error(e);
  showError('asset load failed: ' + String(e));
});
requestAnimationFrame(loop);
