// Illustration library for ZBR-AF1-SW-001.
// Every diagram is drawn as a clean line-art engineering figure with a
// transparent background so it reads as if drafted directly on the page.
// Palette matches the document body.

const fs = require('fs');
const path = require('path');
const { Resvg } = require('@resvg/resvg-js');

const OUT = __dirname;

// ── Palette ──────────────────────────────────────────────────────────────
const INK   = '#25231B';  // primary ink
const MID   = '#8A8172';  // secondary ink (annotations, minor rules)
const RULE  = '#B9B1A1';  // hairline rules
const ACC   = '#B85425';  // burnt-terracotta accent (only for critical marks)
const FAINT = '#DED6C4';  // very light hatching

// ── Common helpers ───────────────────────────────────────────────────────
function svgHead(w, h) {
    // NO background rect: transparent so the page shows through.
    return `<svg viewBox="0 0 ${w} ${h}" xmlns="http://www.w3.org/2000/svg" fill="none">
<defs>
  <pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <line x1="0" y1="0" x2="0" y2="6" stroke="${FAINT}" stroke-width="1"/>
  </pattern>
  <pattern id="dot" width="8" height="8" patternUnits="userSpaceOnUse">
    <circle cx="1" cy="1" r="0.6" fill="${MID}"/>
  </pattern>
</defs>`;
}

function tick(x1, y1, x2, y2, color = INK, w = 1) {
    return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${w}"/>`;
}

// Every figure ends up at ~7 in wide in the docx; at the original size 8-10 pt
// the labels vanished. TXT_SCALE bumps every label so it prints legibly.
const TXT_SCALE = 1.75;
function label(x, y, text, opts = {}) {
    const {
        size = 10, color = INK, anchor = 'start', family = 'Courier New, monospace',
        weight = 'normal', spacing = 0, italic = false
    } = opts;
    const scaled = Math.round(size * TXT_SCALE);
    return `<text x="${x}" y="${y}" font-size="${scaled}" fill="${color}"
        text-anchor="${anchor}" font-family="${family}" font-weight="${weight}"
        letter-spacing="${spacing}" font-style="${italic ? 'italic' : 'normal'}">${text}</text>`;
}

// Callout: a fine leader line from an anchor point to a label, ending in a
// small square terminator. This is the visual grammar the whole document uses.
function callout(ax, ay, lx, ly, text, opts = {}) {
    const { anchor = 'start', size = 10, color = INK, dy = -3 } = opts;
    // corner-elbow leader
    const mx = (ax + lx) / 2;
    const line = `<polyline points="${ax},${ay} ${mx},${ay} ${lx},${ly}"
        stroke="${MID}" stroke-width="0.9" fill="none"/>`;
    const dot = `<circle cx="${ax}" cy="${ay}" r="2" fill="${color}"/>`;
    const term = `<rect x="${lx - 1.5}" y="${ly - 1.5}" width="3" height="3" fill="${color}"/>`;
    return line + dot + term + label(lx + 6, ly + dy + 12, text, { anchor, size, color });
}

// Title block used at the bottom-right of every figure (drafting style).
function titleBlock(fig, name, x, y) {
    const w = 460, h = 130;
    return `
    <g transform="translate(${x},${y})">
      <rect x="0" y="0" width="${w}" height="${h}" fill="none" stroke="${INK}" stroke-width="1"/>
      <line x1="0" y1="30" x2="${w}" y2="30" stroke="${INK}" stroke-width="0.7"/>
      <line x1="0" y1="72" x2="${w}" y2="72" stroke="${INK}" stroke-width="0.7"/>
      <line x1="260" y1="30" x2="260" y2="${h}" stroke="${INK}" stroke-width="0.7"/>
      ${label(10, 22, 'ZEROBIONIC', { size: 9, spacing: 2, weight: 'bold' })}
      ${label(w - 10, 22, 'AF-1 ASSISTIVE ROBOTICS', { size: 8, spacing: 1, anchor: 'end', color: MID })}
      ${label(10, 52, 'TITLE', { size: 6, spacing: 1, color: MID })}
      ${label(10, 66, name.toUpperCase(), { size: 9, spacing: 1, weight: 'bold' })}
      ${label(10, 92, 'DOC', { size: 6, spacing: 1, color: MID })}
      ${label(10, 106, 'ZBR-AF1-SW-001', { size: 8 })}
      ${label(10, 122, 'REV 0.1', { size: 7, color: MID })}
      ${label(270, 52, 'FIGURE', { size: 6, spacing: 1, color: MID })}
      ${label(270, 66, fig, { size: 12, spacing: 2, weight: 'bold' })}
      ${label(270, 92, 'SHEET', { size: 6, spacing: 1, color: MID })}
      ${label(270, 106, '1 OF 1', { size: 8 })}
      ${label(270, 122, 'SCALE N.T.S.', { size: 7, color: MID })}
    </g>`;
}

function borderFrame(w, h) {
    // A single hairline frame + faint corner marks: reads like a drafting sheet
    // without adding a solid box behind the artwork.
    const pad = 22;
    const c = 12;
    return `
      <rect x="${pad}" y="${pad}" width="${w - pad * 2}" height="${h - pad * 2}"
            fill="none" stroke="${RULE}" stroke-width="0.6"/>
      <path d="M ${pad - 4} ${pad + c} L ${pad - 4} ${pad - 4} L ${pad + c} ${pad - 4}" stroke="${INK}" stroke-width="0.9" fill="none"/>
      <path d="M ${w - pad + 4} ${pad + c} L ${w - pad + 4} ${pad - 4} L ${w - pad - c} ${pad - 4}" stroke="${INK}" stroke-width="0.9" fill="none"/>
      <path d="M ${pad - 4} ${h - pad - c} L ${pad - 4} ${h - pad + 4} L ${pad + c} ${h - pad + 4}" stroke="${INK}" stroke-width="0.9" fill="none"/>
      <path d="M ${w - pad + 4} ${h - pad - c} L ${w - pad + 4} ${h - pad + 4} L ${w - pad - c} ${h - pad + 4}" stroke="${INK}" stroke-width="0.9" fill="none"/>
    `;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 01  ·  AF-1 HEAD, ANATOMICAL CUTAWAY
// The hero illustration. A stylised profile of the AF-1 head drawn as a
// mechanical cutaway. Every callout maps an internal component to the
// software function it implements.
// ═══════════════════════════════════════════════════════════════════════════
function figHeadCutaway() {
    // AF-1 industrial head, front elevation, drawn to match the unit on the cover.
    // Cutaway on the right half reveals the internal software components.
    const W = 1500, H = 900;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    const cx = 520, cy = 460;

    // ── Outer sensor-block head (rectangular, chamfered corners) ────────────
    const headW = 320, headH = 240;
    const hL = cx - headW / 2, hR = cx + headW / 2;
    const hT = cy - headH / 2, hB = cy + headH / 2;
    const ch = 22; // chamfer

    s += `
      <!-- crown assembly (small block on top) -->
      <rect x="${cx - 30}" y="${hT - 34}" width="60" height="34" fill="none" stroke="${INK}" stroke-width="2"/>
      <rect x="${cx - 22}" y="${hT - 44}" width="44" height="14" fill="none" stroke="${INK}" stroke-width="1.4"/>
      <line x1="${cx - 22}" y1="${hT - 22}" x2="${cx + 22}" y2="${hT - 22}" stroke="${INK}" stroke-width="0.9"/>

      <!-- main sensor block (head outer) with chamfered corners -->
      <path d="
        M ${hL + ch} ${hT}
        L ${hR - ch} ${hT}
        L ${hR} ${hT + ch}
        L ${hR} ${hB - ch}
        L ${hR - ch} ${hB}
        L ${hL + ch} ${hB}
        L ${hL} ${hB - ch}
        L ${hL} ${hT + ch}
        Z" fill="none" stroke="${INK}" stroke-width="2.4"/>

      <!-- panel seam lines -->
      <line x1="${hL + 16}" y1="${hT + 16}" x2="${hR - 16}" y2="${hT + 16}" stroke="${INK}" stroke-width="0.8"/>
      <line x1="${hL + 16}" y1="${hB - 16}" x2="${hR - 16}" y2="${hB - 16}" stroke="${INK}" stroke-width="0.8"/>

      <!-- visor slot (horizontal aperture housing the cameras) -->
      <rect x="${cx - 130}" y="${cy - 40}" width="260" height="50" rx="8" fill="none" stroke="${INK}" stroke-width="2"/>
      <rect x="${cx - 122}" y="${cy - 32}" width="244" height="34" rx="4" fill="url(#hatch)" stroke="${INK}" stroke-width="1"/>

      <!-- two camera lenses inside the visor -->
      <circle cx="${cx - 80}" cy="${cy - 15}" r="14" fill="none" stroke="${INK}" stroke-width="1.6"/>
      <circle cx="${cx - 80}" cy="${cy - 15}" r="8"  fill="${INK}"/>
      <circle cx="${cx - 80}" cy="${cy - 15}" r="3"  fill="#fff"/>
      <circle cx="${cx + 80}" cy="${cy - 15}" r="14" fill="none" stroke="${INK}" stroke-width="1.6"/>
      <circle cx="${cx + 80}" cy="${cy - 15}" r="8"  fill="${INK}"/>
      <circle cx="${cx + 80}" cy="${cy - 15}" r="3"  fill="#fff"/>

      <!-- mic grille (row of vertical vents, lower face) -->
      ${Array.from({length: 14}, (_, i) => `<line x1="${cx - 84 + i * 12}" y1="${cy + 40}" x2="${cx - 84 + i * 12}" y2="${cy + 68}" stroke="${INK}" stroke-width="1.4"/>`).join('')}
      <rect x="${cx - 90}" y="${cy + 36}" width="180" height="38" fill="none" stroke="${INK}" stroke-width="1.2"/>

      <!-- corner bolts -->
      ${[[hL + 12, hT + 12], [hR - 12, hT + 12], [hL + 12, hB - 12], [hR - 12, hB - 12]].map(p =>
        `<circle cx="${p[0]}" cy="${p[1]}" r="3.5" fill="none" stroke="${INK}" stroke-width="1"/>
         <circle cx="${p[0]}" cy="${p[1]}" r="1"   fill="${INK}"/>`).join('')}
    `;

    // ── Shoulders / neck stub (hint of the body from the cover) ────────────
    s += `
      <!-- neck collar -->
      <rect x="${cx - 60}" y="${hB}" width="120" height="26" fill="none" stroke="${INK}" stroke-width="2"/>
      <line x1="${cx - 60}" y1="${hB + 12}" x2="${cx + 60}" y2="${hB + 12}" stroke="${INK}" stroke-width="0.8"/>

      <!-- pauldron shoulder joints (round) -->
      <circle cx="${cx - 140}" cy="${hB + 60}" r="34" fill="none" stroke="${INK}" stroke-width="2"/>
      <circle cx="${cx - 140}" cy="${hB + 60}" r="10" fill="none" stroke="${INK}" stroke-width="1"/>
      ${Array.from({length: 8}, (_, i) => {
        const a = i * Math.PI / 4;
        return `<circle cx="${cx - 140 + Math.cos(a) * 22}" cy="${hB + 60 + Math.sin(a) * 22}" r="2" fill="${INK}"/>`;
      }).join('')}
      <circle cx="${cx + 140}" cy="${hB + 60}" r="34" fill="none" stroke="${INK}" stroke-width="2"/>
      <circle cx="${cx + 140}" cy="${hB + 60}" r="10" fill="none" stroke="${INK}" stroke-width="1"/>
      ${Array.from({length: 8}, (_, i) => {
        const a = i * Math.PI / 4;
        return `<circle cx="${cx + 140 + Math.cos(a) * 22}" cy="${hB + 60 + Math.sin(a) * 22}" r="2" fill="${INK}"/>`;
      }).join('')}
    `;

    // ── Cutaway edge (broken zigzag on the right half of the head) ─────────
    s += `
      <path d="
        M ${cx} ${hT - 44}
        L ${cx + 4} ${hT - 20}
        L ${cx - 4} ${hT + 10}
        L ${cx + 6} ${hT + 40}
        L ${cx - 4} ${cy - 32}
        L ${cx + 6} ${cy - 15}
        L ${cx - 4} ${cy + 6}
        L ${cx + 6} ${cy + 40}
        L ${cx - 4} ${cy + 68}
        L ${cx + 6} ${hB - 16}
        L ${cx - 4} ${hB}
        L ${cx + 4} ${hB + 26}
      " fill="none" stroke="${INK}" stroke-width="1.2" stroke-dasharray="5 3"/>
    `;

    // ── Interior components revealed by the cutaway (right side) ───────────
    // 1. CORE (central processing core, sits behind the visor)
    // 2. SPINE (data conduit descending from crown to memory)
    // 3. MEMORY chambers (stack of plates in the lower jaw area)
    // 4. Audio bus lines from mic to core, vision bus from cameras to core,
    //    reply bus from core down to speaker.
    s += `
      <!-- CORE: octagonal processor package -->
      <g transform="translate(${cx + 60},${cy - 40})">
        <polygon points="-40,-18 -18,-40 18,-40 40,-18 40,18 18,40 -18,40 -40,18"
                 fill="none" stroke="${INK}" stroke-width="1.8"/>
        <polygon points="-30,-14 -14,-30 14,-30 30,-14 30,14 14,30 -14,30 -30,14"
                 fill="url(#hatch)" stroke="${INK}" stroke-width="1"/>
        <rect x="-8" y="-8" width="16" height="16" fill="${INK}"/>
      </g>

      <!-- SPINE: two parallel conductors descending from crown to memory -->
      <path d="M ${cx + 18} ${hT - 30} L ${cx + 18} ${hB + 12}" stroke="${INK}" stroke-width="1.4"/>
      <path d="M ${cx + 28} ${hT - 30} L ${cx + 28} ${hB + 12}" stroke="${INK}" stroke-width="1.4"/>
      ${Array.from({length: 10}, (_, i) => `<line x1="${cx + 18}" y1="${hT - 20 + i * 24}" x2="${cx + 28}" y2="${hT - 20 + i * 24}" stroke="${INK}" stroke-width="0.7"/>`).join('')}

      <!-- MEMORY chambers: stack of six labelled plates in the lower jaw area -->
      ${Array.from({length: 6}, (_, i) => {
        const y = cy + 85 + i * 9;
        return `<rect x="${cx + 44}" y="${y}" width="80" height="6" fill="none" stroke="${INK}" stroke-width="0.9"/>
                <circle cx="${cx + 118}" cy="${y + 3}" r="1.5" fill="${INK}"/>`;
      }).join('')}

      <!-- Circuit-board texture backing -->
      <rect x="${cx + 128}" y="${cy + 82}" width="10" height="60" fill="url(#dot)" opacity="0.8"/>

      <!-- Audio bus: mic grille to core -->
      <path d="M ${cx + 40} ${cy + 55} C ${cx + 55} ${cy + 30}, ${cx + 70} ${cy}, ${cx + 65} ${cy - 20}"
            fill="none" stroke="${MID}" stroke-width="0.9"/>
      <!-- Vision bus: right camera to core -->
      <path d="M ${cx + 80} ${cy - 15} C ${cx + 90} ${cy - 25}, ${cx + 85} ${cy - 32}, ${cx + 78} ${cy - 38}"
            fill="none" stroke="${MID}" stroke-width="0.9"/>
      <!-- Reply bus: core to memory / speaker -->
      <path d="M ${cx + 55} ${cy - 5} C ${cx + 65} ${cy + 30}, ${cx + 55} ${cy + 70}, ${cx + 60} ${cy + 90}"
            fill="none" stroke="${MID}" stroke-width="0.9"/>
    `;

    // ── Callouts (lots of room now — text at size 11-12 scales to ~19-21 pt) ─
    const callouts = `
      ${callout(cx + 60, cy - 40, W - 480, 160, 'CENTRAL CORE', { color: INK, size: 12, weight: 'bold' })}
      ${label(W - 460, 178, 'the language model, persona, and', { size: 10, color: INK })}
      ${label(W - 460, 194, 'streaming generator. The "brain".', { size: 10, color: INK })}

      ${callout(cx + 23, hB + 12, W - 480, 260, 'SPINE  ·  SSH TUNNEL', { color: ACC, size: 12, weight: 'bold' })}
      ${label(W - 460, 278, 'the single conduit to the GPU node,', { size: 10, color: ACC })}
      ${label(W - 460, 294, 'five forwarded ports, one sheath.', { size: 10, color: ACC })}

      ${callout(cx, hT - 44, W - 480, 360, 'CROWN ANTENNA', { color: INK, size: 12, weight: 'bold' })}
      ${label(W - 460, 378, 'wake-word ingest, always-on 30 ms', { size: 10, color: INK })}
      ${label(W - 460, 394, 'frames on the microphone stream.', { size: 10, color: INK })}

      ${callout(cx - 80, cy - 15, 60, 200, 'LEFT CAMERA', { color: INK, size: 12, weight: 'bold' })}
      ${label(60, 218, 'open-vocab YOLO, colour naming,', { size: 10, color: INK })}
      ${label(60, 234, 'motion Tier 0 (every frame).', { size: 10, color: INK })}

      ${callout(cx + 80, cy - 15, 60, 320, 'RIGHT CAMERA', { color: INK, size: 12, weight: 'bold' })}
      ${label(60, 338, 'face embedder (ArcFace 512-d),', { size: 10, color: INK })}
      ${label(60, 354, 'minimum face size 60 px.', { size: 10, color: INK })}

      ${callout(cx, cy + 55, 60, 440, 'MIC GRILLE', { color: INK, size: 12, weight: 'bold' })}
      ${label(60, 458, 'PortAudio 16 kHz mono, gain 6.0,', { size: 10, color: INK })}
      ${label(60, 474, 'ECAPA voice ID off the same stream.', { size: 10, color: INK })}

      ${callout(cx + 85, cy + 118, W - 480, 560, 'MEMORY CHAMBERS', { color: INK, size: 12, weight: 'bold' })}
      ${label(W - 460, 578, 'six SQLite stores plus corpus JSONL:', { size: 10, color: INK })}
      ${label(W - 460, 594, 'memory, identity, guests, episodes,', { size: 10, color: INK })}
      ${label(W - 460, 610, 'objects, curiosity.', { size: 10, color: INK })}

      ${callout(cx - 140, hB + 60, 60, 660, 'PAULDRON SHOULDER', { color: MID, size: 11, weight: 'bold' })}
      ${label(60, 678, 'motion domain of the AF-1 chassis.', { size: 10, color: MID, italic: true })}
      ${label(60, 692, 'not covered in this document.', { size: 9, color: MID, italic: true })}
    `;

    s += callouts;

    // Scale strip
    const stripY = H - 150;
    s += `
      <line x1="60" y1="${stripY}" x2="260" y2="${stripY}" stroke="${INK}" stroke-width="1"/>
      <line x1="60" y1="${stripY - 5}" x2="60" y2="${stripY + 5}" stroke="${INK}" stroke-width="1"/>
      <line x1="260" y1="${stripY - 5}" x2="260" y2="${stripY + 5}" stroke="${INK}" stroke-width="1"/>
      ${label(160, stripY + 20, 'SECTION A—A  ·  MEDIAN CORONAL', { size: 9, anchor: 'middle', color: MID, spacing: 2 })}
      ${label(160, stripY + 36, 'CUTAWAY REVEALS INTERIOR ALONG THE RIGHT HALF', { size: 8, anchor: 'middle', color: MID, spacing: 1 })}
    `;

    s += titleBlock('1', 'AF-1 HEAD  ·  ANATOMICAL CUTAWAY', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 02  ·  TWO-MACHINE NERVOUS SYSTEM
// AF-1 head on the left, GPU node on the right, joined by the SSH tunnel
// drawn as a nerve fiber. Not a block diagram: an anatomical schematic.
// ═══════════════════════════════════════════════════════════════════════════
function figNervousSystem() {
    const W = 1200, H = 700;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    // AF-1 head (industrial sensor block, matches the cover) on the left
    const cx = 240, cy = 340;
    const hw = 200, hh = 150;
    const ch = 14;
    const hL = cx - hw / 2, hR = cx + hw / 2, hT = cy - hh / 2, hB = cy + hh / 2;
    s += `
      <!-- crown -->
      <rect x="${cx - 22}" y="${hT - 24}" width="44" height="24" fill="none" stroke="${INK}" stroke-width="1.6"/>
      <rect x="${cx - 16}" y="${hT - 30}" width="32" height="8"  fill="none" stroke="${INK}" stroke-width="1.2"/>
      <!-- sensor block -->
      <path d="
        M ${hL + ch} ${hT} L ${hR - ch} ${hT} L ${hR} ${hT + ch}
        L ${hR} ${hB - ch} L ${hR - ch} ${hB} L ${hL + ch} ${hB}
        L ${hL} ${hB - ch} L ${hL} ${hT + ch} Z
      " fill="none" stroke="${INK}" stroke-width="2"/>
      <!-- visor slot -->
      <rect x="${cx - 84}" y="${cy - 20}" width="168" height="30" rx="6" fill="url(#hatch)" stroke="${INK}" stroke-width="1.4"/>
      <circle cx="${cx - 48}" cy="${cy - 5}" r="8"  fill="none" stroke="${INK}" stroke-width="1.2"/>
      <circle cx="${cx - 48}" cy="${cy - 5}" r="4"  fill="${INK}"/>
      <circle cx="${cx + 48}" cy="${cy - 5}" r="8"  fill="none" stroke="${INK}" stroke-width="1.2"/>
      <circle cx="${cx + 48}" cy="${cy - 5}" r="4"  fill="${INK}"/>
      <!-- mic grille -->
      ${Array.from({length: 10}, (_, i) => `<line x1="${cx - 54 + i * 12}" y1="${cy + 30}" x2="${cx - 54 + i * 12}" y2="${cy + 52}" stroke="${INK}" stroke-width="1.1"/>`).join('')}
      <!-- neck stub -->
      <rect x="${cx - 40}" y="${hB}" width="80" height="18" fill="none" stroke="${INK}" stroke-width="1.6"/>
      ${label(cx, hB + 46, 'AF-1 HEAD  ·  RASPBERRY PI 5', { size: 11, anchor: 'middle', spacing: 2, weight: 'bold' })}
      ${label(cx, hB + 64, 'real-time capture, wake, VAD, loop', { size: 9, anchor: 'middle', color: MID, italic: true })}
    `;

    // GPU node — drawn as a stack of rack-mounted server slices
    const gx = 960, gy = 330;
    s += `
      <!-- rack shell -->
      <rect x="${gx - 90}" y="${gy - 80}" width="180" height="220" fill="none" stroke="${INK}" stroke-width="1.6"/>
      <!-- five slots for five servers -->
      ${['Whisper', 'Orpheus', 'Vision', 'Ollama', 'SearXNG'].map((n, i) => {
        const y = gy - 68 + i * 42;
        return `
          <rect x="${gx - 80}" y="${y}" width="160" height="34" fill="none" stroke="${INK}" stroke-width="1"/>
          <circle cx="${gx - 72}" cy="${y + 17}" r="3" fill="${ACC}"/>
          ${label(gx - 62, y + 20, n, { size: 9, weight: 'bold' })}
          ${label(gx + 72, y + 20, [':9000', ':9100', ':8000', ':11434', ':8080'][i], { size: 8, anchor: 'end', color: MID })}
          <!-- fan slats -->
          ${Array.from({length: 6}, (_, k) => `<line x1="${gx + 10 + k * 4}" y1="${y + 6}" x2="${gx + 10 + k * 4}" y2="${y + 28}" stroke="${MID}" stroke-width="0.4"/>`).join('')}
        `;
      }).join('')}
      ${label(gx, gy + 158, 'GPU NODE  ·  five model servers', { size: 9, anchor: 'middle', spacing: 1 })}
      ${label(gx, gy + 171, 'each restartable on its own', { size: 8, anchor: 'middle', color: MID })}
    `;

    // Nerve fiber (the SSH tunnel) — drawn as a bundle of five conductors,
    // one darker one (Ollama) shown swapping numbers mid-flight.
    const y0 = 340;
    const cx1 = hR, cx2 = gx - 90;
    for (let i = 0; i < 5; i++) {
        const yy = y0 - 40 + i * 20;
        const isOllama = i === 4;
        const stroke = isOllama ? ACC : INK;
        const w = isOllama ? 1.6 : 1;
        s += `<path d="M ${cx1} ${yy} C ${cx1 + 100} ${yy - 10}, ${cx2 - 100} ${yy + 10}, ${cx2} ${yy}"
                fill="none" stroke="${stroke}" stroke-width="${w}"/>`;
        // Label each nerve
        const labels = [':9000 whisper', ':9100 orpheus', ':8000 vision', ':8080 searxng', ':11435 → :11434 ollama'];
        if (i < 4) {
          s += label((cx1 + cx2) / 2, yy - 4, labels[i], { size: 7, anchor: 'middle', color: MID });
        } else {
          s += label((cx1 + cx2) / 2, yy + 12, labels[i], { size: 7, anchor: 'middle', color: ACC, weight: 'bold' });
        }
    }

    // Sheath (single tunnel wraps all five)
    s += `
      <path d="M ${cx1 - 6} ${y0 - 46} C ${cx1 + 110} ${y0 - 56}, ${cx2 - 110} ${y0 - 36}, ${cx2 + 6} ${y0 - 46}"
            fill="none" stroke="${MID}" stroke-width="0.6" stroke-dasharray="4 3"/>
      <path d="M ${cx1 - 6} ${y0 + 46} C ${cx1 + 110} ${y0 + 36}, ${cx2 - 110} ${y0 + 56}, ${cx2 + 6} ${y0 + 46}"
            fill="none" stroke="${MID}" stroke-width="0.6" stroke-dasharray="4 3"/>
      ${label((cx1 + cx2) / 2, y0 - 62, 'SSH TUNNEL (autossh) · single sheath, five conductors', { size: 8, anchor: 'middle', color: MID, spacing: 1 })}
    `;

    // Callouts explaining the important behaviours
    s += callout(cx1 + 10, y0 + 40, 320, H - 130,
      'nothing here is exposed to the internet;', { color: INK, size: 9 });
    s += label(326, H - 118, 'a stall is rebuilt inside ~45 s by autossh.', { size: 9, color: INK });

    s += callout(cx2 - 10, y0 + 40, 320, H - 100,
      'ollama alone shifts number in flight, so a', { color: ACC, size: 9 });
    s += label(326, H - 88, 'local pi ollama can keep 11434 for itself.', { size: 9, color: ACC });

    s += titleBlock('2', 'TWO-MACHINE NERVOUS SYSTEM', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 03  ·  PROCESS SET AS A HIVE
// The process set as concentric rings around the main loop, each thread
// drawn as an annotated cell so you can see the shape of the concurrency.
// ═══════════════════════════════════════════════════════════════════════════
function figProcessHive() {
    // Rebuilt with much larger canvas, larger circles, and descriptions
    // placed cleanly OUTSIDE each circle so nothing overlaps.
    const W = 1500, H = 1200;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    const cx = W / 2, cy = 560;

    // Main loop at centre
    s += `
      <circle cx="${cx}" cy="${cy}" r="100" fill="none" stroke="${INK}" stroke-width="2.4"/>
      <circle cx="${cx}" cy="${cy}" r="78" fill="url(#hatch)"/>
      ${label(cx, cy - 8, 'MAIN', { size: 16, anchor: 'middle', weight: 'bold' })}
      ${label(cx, cy + 16, 'LOOP', { size: 16, anchor: 'middle', weight: 'bold' })}
      ${label(cx, cy + 44, 'four states, one thread', { size: 8, anchor: 'middle', color: MID, italic: true })}
    `;

    // Persistent threads: ring 1, drawn as bigger cells with the label INSIDE
    // and the description on a leader line OUTSIDE the circle.
    const persistent = [
        ['MIC',    'PortAudio 30 ms frames'],
        ['CAM',    'CameraStream grab'],
        ['EYES',   'tiered perception'],
        ['NARR',   'Tier 2 VLM narrator'],
        ['SURP',   'prediction-error gate'],
        ['PROAC',  'presence, curiosity, remarks'],
        ['CTRL',   'HTTP control plane :8090'],
        ['PREV',   'MJPG preview :8008'],
    ];
    const ringR = 280;
    persistent.forEach((p, i) => {
        const a = (i / persistent.length) * Math.PI * 2 - Math.PI / 2;
        const x = cx + Math.cos(a) * ringR;
        const y = cy + Math.sin(a) * ringR;
        // Cell
        s += `<circle cx="${x}" cy="${y}" r="42" fill="none" stroke="${INK}" stroke-width="1.8"/>`;
        s += label(x, y + 4, p[0], { size: 12, anchor: 'middle', weight: 'bold' });
        // Leader out to a description tag OUTSIDE the ring
        const lx = cx + Math.cos(a) * (ringR + 130);
        const ly = cy + Math.sin(a) * (ringR + 130);
        s += `<line x1="${x + Math.cos(a) * 42}" y1="${y + Math.sin(a) * 42}"
                    x2="${lx}" y2="${ly}" stroke="${MID}" stroke-width="0.7"/>`;
        // Anchor left/middle/right based on which side of the ring the tag is on
        const anchor = Math.cos(a) > 0.3 ? 'start' : Math.cos(a) < -0.3 ? 'end' : 'middle';
        const dx = anchor === 'start' ? 8 : anchor === 'end' ? -8 : 0;
        s += label(lx + dx, ly + 4, p[1], { size: 9, anchor, color: INK });
        // Spoke back to the main loop
        s += `<line x1="${cx + Math.cos(a) * 100}" y1="${cy + Math.sin(a) * 100}"
                    x2="${x - Math.cos(a) * 42}" y2="${y - Math.sin(a) * 42}"
                    stroke="${INK}" stroke-width="0.6"/>`;
    });

    // Transient threads: an inner band between the loop and the persistent ring,
    // drawn dashed. Names only (no crowded descriptions) plus a caption list
    // at the bottom naming what each transient does.
    const transient = [
        ['stt-spec', 190, -Math.PI / 2 + Math.PI / 8],
        ['stt',      190, -Math.PI / 2 + Math.PI / 8 + Math.PI / 4],
        ['recall',   190, -Math.PI / 2 + Math.PI / 8 + 2 * Math.PI / 4],
        ['llm',      190, -Math.PI / 2 + Math.PI / 8 + 3 * Math.PI / 4],
        ['tts',      190, -Math.PI / 2 + Math.PI / 8 + 4 * Math.PI / 4],
        ['barge',    190, -Math.PI / 2 + Math.PI / 8 + 5 * Math.PI / 4],
        ['compact',  190, -Math.PI / 2 + Math.PI / 8 + 6 * Math.PI / 4],
        ['save',     190, -Math.PI / 2 + Math.PI / 8 + 7 * Math.PI / 4],
    ];
    transient.forEach(t => {
        const [name, r, a] = t;
        const x = cx + Math.cos(a) * r;
        const y = cy + Math.sin(a) * r;
        s += `<circle cx="${x}" cy="${y}" r="30" fill="#fff" stroke="${INK}" stroke-width="1.2" stroke-dasharray="3 2"/>`;
        s += label(x, y + 4, name, { size: 10, anchor: 'middle', weight: 'bold' });
    });

    // Legend and transient descriptions at the bottom
    const ly = H - 250;
    s += `<line x1="80" y1="${ly - 14}" x2="${W - 80}" y2="${ly - 14}" stroke="${RULE}" stroke-width="0.8"/>`;
    s += label(80, ly, 'LEGEND', { size: 9, spacing: 3, weight: 'bold' });
    s += `<circle cx="180" cy="${ly + 22}" r="12" fill="none" stroke="${INK}" stroke-width="1.6"/>`;
    s += label(200, ly + 28, 'PERSISTENT thread  ·  starts once, lives with the process', { size: 10 });
    s += `<circle cx="180" cy="${ly + 52}" r="12" fill="#fff" stroke="${INK}" stroke-width="1.2" stroke-dasharray="3 2"/>`;
    s += label(200, ly + 58, 'TRANSIENT thread  ·  born and dies inside one turn', { size: 10 });
    s += `<circle cx="180" cy="${ly + 82}" r="12" fill="url(#hatch)" stroke="${INK}" stroke-width="1.6"/>`;
    s += label(200, ly + 88, 'MAIN LOOP  ·  the only thread that transitions the state machine', { size: 10 });

    // Transient descriptions as a two column key on the right of the legend
    const kx = 780;
    s += label(kx, ly, 'TRANSIENT ROLES', { size: 9, spacing: 3, weight: 'bold' });
    const roles = [
        ['stt-spec', 'speculative transcription'],
        ['stt',      'final transcription'],
        ['recall',   'memory relevance search'],
        ['llm',      'streaming generation'],
        ['tts',      'sentence-by-sentence synth'],
        ['barge',    'interrupt monitor'],
        ['compact',  'rolling summary fold'],
        ['save',     'per-speaker durable write'],
    ];
    roles.forEach((r, i) => {
        const col = i % 2;
        const row = Math.floor(i / 2);
        s += label(kx + col * 320, ly + 22 + row * 20, r[0], { size: 9, weight: 'bold' });
        s += label(kx + 90 + col * 320, ly + 22 + row * 20, r[1], { size: 9, color: MID });
    });

    s += titleBlock('3', 'THREADS AS A HIVE AROUND THE LOOP', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 04  ·  TURN TIMING TRACE
// Logic-analyzer style, but drawn like a real chart-recorder trace with
// pen marks rather than a filled bar chart.
// ═══════════════════════════════════════════════════════════════════════════
function figTurnTiming() {
    const W = 1000, H = 640;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    const x0 = 190, x1 = W - 60;
    const totalMs = 4400;
    const scale = (x1 - x0) / totalMs;

    const lanes = [
        ['MIC CAPTURE',      0, 4400, 'persistent'],
        ['ENDPOINTER',       0, 2400, ''],
        ['STT-SPEC',       1600, 800,  ''],
        ['STT FINAL',      2400, 700,  ''],
        ['IDENTITY',       2400, 300,  ''],
        ['DIARIZE',        2400, 250,  ''],
        ['AFFECT',         2400, 220,  ''],
        ['RECALL',         2400, 280,  ''],
        ['LLM STREAM',     2700, 1400, ''],
        ['FILLER AUDIO',   3050, 350,  ''],
        ['TTS PRODUCER',   3100, 1200, ''],
        ['REPLY AUDIO',    3400, 1000, 'critical'],
        ['BARGEIN WATCH',  3400, 1000, ''],
    ];

    // Axis ticks (silence-axis on the bottom)
    for (let ms = 0; ms <= totalMs; ms += 200) {
        const x = x0 + ms * scale;
        const major = ms % 1000 === 0;
        s += `<line x1="${x}" y1="${H - 130}" x2="${x}" y2="${H - 130 + (major ? 8 : 4)}" stroke="${INK}" stroke-width="${major ? 1 : 0.5}"/>`;
        if (major) s += label(x, H - 115, `${ms / 1000}s`, { size: 8, anchor: 'middle', color: MID });
    }
    s += tick(x0, H - 130, x1, H - 130, INK, 1);
    s += label((x0 + x1) / 2, H - 100, 'TIME FROM WAKE-WORD ACCEPT', { size: 8, anchor: 'middle', color: MID, spacing: 2 });

    // Phase bands at the top
    const phases = [
        [0, 1600, 'LISTENING'],
        [1600, 2400, 'PAUSE (silence wait)'],
        [2400, 3400, 'THINKING'],
        [3400, 4400, 'SPEAKING'],
    ];
    phases.forEach((p, i) => {
        const [a, b, name] = p;
        const px0 = x0 + a * scale;
        const px1 = x0 + b * scale;
        s += `<line x1="${px0}" y1="80" x2="${px1}" y2="80" stroke="${INK}" stroke-width="1.2"/>`;
        s += `<line x1="${px0}" y1="76" x2="${px0}" y2="84" stroke="${INK}" stroke-width="1.2"/>`;
        s += `<line x1="${px1}" y1="76" x2="${px1}" y2="84" stroke="${INK}" stroke-width="1.2"/>`;
        s += label((px0 + px1) / 2, 72, name, { size: 8, anchor: 'middle', color: INK, spacing: 1, weight: 'bold' });
    });

    // Lanes
    const laneH = 30;
    const laneY0 = 110;
    lanes.forEach((l, i) => {
        const [name, start, dur, tag] = l;
        const y = laneY0 + i * laneH;
        // Lane baseline
        s += `<line x1="${x0 - 10}" y1="${y + laneH / 2}" x2="${x1}" y2="${y + laneH / 2}" stroke="${RULE}" stroke-width="0.4" stroke-dasharray="1 3"/>`;
        s += label(x0 - 20, y + laneH / 2 + 4, name, { size: 8, anchor: 'end', color: INK });
        // Activity band drawn as a hand-inked mark
        const px = x0 + start * scale;
        const pw = dur * scale;
        const color = tag === 'critical' ? ACC : INK;
        const th = tag === 'critical' ? 3 : 2;
        s += `<line x1="${px}" y1="${y + laneH / 2}" x2="${px + pw}" y2="${y + laneH / 2}" stroke="${color}" stroke-width="${th}"/>`;
        s += `<line x1="${px}" y1="${y + laneH / 2 - 5}" x2="${px}" y2="${y + laneH / 2 + 5}" stroke="${color}" stroke-width="${th}"/>`;
        s += `<line x1="${px + pw}" y1="${y + laneH / 2 - 5}" x2="${px + pw}" y2="${y + laneH / 2 + 5}" stroke="${color}" stroke-width="${th}"/>`;
    });

    // Key markers
    const mark = (ms, label_) => {
        const x = x0 + ms * scale;
        s += `<line x1="${x}" y1="80" x2="${x}" y2="${H - 130}" stroke="${ACC}" stroke-width="0.6" stroke-dasharray="4 3"/>`;
        s += label(x, H - 82, label_, { size: 7, anchor: 'middle', color: ACC, spacing: 1 });
    };
    mark(1600, 'SPEECH ENDS');
    mark(2400, 'ENDPOINT CONFIRMED');
    mark(3400, 'FIRST REPLY AUDIO');

    // Legend
    s += `<line x1="60" y1="${H - 60}" x2="90" y2="${H - 60}" stroke="${INK}" stroke-width="2"/>`;
    s += label(96, H - 56, 'THREAD ACTIVE', { size: 8, color: INK });
    s += `<line x1="220" y1="${H - 60}" x2="250" y2="${H - 60}" stroke="${ACC}" stroke-width="3"/>`;
    s += label(256, H - 56, 'ON THE CRITICAL PATH TO FIRST AUDIO', { size: 8, color: ACC });

    s += titleBlock('4', 'TURN TIMING TRACE', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 05  ·  PERCEPTION TIERS AS A RETICULAR STACK
// ═══════════════════════════════════════════════════════════════════════════
function figTiers() {
    const W = 1000, H = 520;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    const x0 = 130, x1 = W - 80;
    const t0 = 12; // 12 seconds
    const scale = (x1 - x0) / t0;

    // Motion band (shaded)
    const mStart = 0, mEnd = 6;
    s += `<rect x="${x0 + mStart * scale}" y="90" width="${(mEnd - mStart) * scale}" height="330"
              fill="url(#hatch)" stroke="none"/>`;
    s += label(x0 + ((mStart + mEnd) / 2) * scale, 82, 'MOTION ACTIVE', { size: 9, anchor: 'middle', color: MID, spacing: 2 });

    // Linger extension
    const lEnd = 9;
    s += `<rect x="${x0 + mEnd * scale}" y="90" width="${(lEnd - mEnd) * scale}" height="330"
              fill="none" stroke="${MID}" stroke-width="0.6" stroke-dasharray="3 3"/>`;
    s += label(x0 + ((mEnd + lEnd) / 2) * scale, 82, 'LINGER 3.0 s', { size: 8, anchor: 'middle', color: MID });

    // Three tier lanes
    const tiers = [
        { name: 'TIER 0  ·  MOTION', y: 140, ticks: Array.from({ length: 72 }, (_, i) => i * (t0 / 72)) },
        { name: 'TIER 1  ·  DETECTION', y: 240,
          ticks: [
              ...Array.from({ length: Math.round((mEnd - mStart) / 0.05) }, (_, i) => mStart + i * 0.05),
              ...Array.from({ length: Math.round((lEnd - mEnd) / 0.05) }, (_, i) => mEnd + i * 0.05),
              ...Array.from({ length: Math.floor((t0 - lEnd) / 2) }, (_, i) => lEnd + i * 2 + 2),
          ]
        },
        { name: 'TIER 2  ·  NARRATION', y: 340,
          ticks: Array.from({ length: Math.floor(t0 / 3) }, (_, i) => (i + 1) * 3)
        },
    ];

    tiers.forEach(t => {
        s += `<line x1="${x0}" y1="${t.y}" x2="${x1}" y2="${t.y}" stroke="${INK}" stroke-width="1"/>`;
        s += label(x0 - 12, t.y + 4, t.name, { size: 9, anchor: 'end', spacing: 1, weight: 'bold' });
        t.ticks.forEach(sec => {
            if (sec > t0) return;
            const xx = x0 + sec * scale;
            s += `<line x1="${xx}" y1="${t.y - 8}" x2="${xx}" y2="${t.y + 8}" stroke="${INK}" stroke-width="1"/>`;
        });
    });

    // Time axis
    for (let sec = 0; sec <= t0; sec++) {
        const xx = x0 + sec * scale;
        s += `<line x1="${xx}" y1="410" x2="${xx}" y2="418" stroke="${INK}" stroke-width="0.8"/>`;
        s += label(xx, 432, `${sec}s`, { size: 8, anchor: 'middle', color: MID });
    }
    s += tick(x0, 410, x1, 410, INK, 1);

    // Duty ceiling meter on the right
    const meterX = W - 60;
    s += `<rect x="${meterX}" y="140" width="18" height="200" fill="none" stroke="${INK}" stroke-width="1"/>`;
    s += `<rect x="${meterX}" y="${140 + 200 * 0.4}" width="18" height="${200 * 0.6}" fill="${INK}" opacity="0.15"/>`;
    s += `<line x1="${meterX - 4}" y1="${140 + 200 * 0.4}" x2="${meterX + 22}" y2="${140 + 200 * 0.4}" stroke="${ACC}" stroke-width="1.4"/>`;
    s += label(meterX + 26, 140 + 200 * 0.4 + 3, '60 % DUTY CEILING', { size: 8, color: ACC, weight: 'bold' });
    s += label(meterX + 9, 132, 'BUDGET', { size: 7, anchor: 'middle', color: MID, spacing: 1 });

    s += titleBlock('5', 'PERCEPTION TIERS OVER 12 s', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 06  ·  FALLBACK CHANGEOVER SCHEMATIC
// ═══════════════════════════════════════════════════════════════════════════
function figFallback() {
    const W = 1000, H = 580;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    const rows = [
        ['STT',            'Whisper server',   'whisper.cpp on Pi'],
        ['TTS',            'Orpheus vLLM',     'Piper local voice'],
        ['DETECTOR',       'server YOLO11x',   'local YOLOv8-world'],
        ['FACE EMBED',     'server ArcFace',   'local ArcFace ONNX'],
        ['SPEAKER EMBED',  'server ECAPA',     'local ECAPA ONNX'],
        ['OBJECT EMBED',   'server CLIP',      'local histogram'],
    ];

    const leftRail = 250, rightRail = 720;
    const busY = 480;
    const rowH = 58;

    // Rails
    s += `<line x1="${leftRail}" y1="120" x2="${leftRail}" y2="${120 + rows.length * rowH - 20}" stroke="${INK}" stroke-width="2"/>`;
    s += `<line x1="${rightRail}" y1="120" x2="${rightRail}" y2="${120 + rows.length * rowH - 20}" stroke="${INK}" stroke-width="2"/>`;
    s += label(leftRail, 108, 'TUNNEL RAIL (GPU)', { size: 9, anchor: 'middle', spacing: 1, weight: 'bold' });
    s += label(rightRail, 108, 'PI RAIL (LOCAL FALLBACK)', { size: 9, anchor: 'middle', spacing: 1, weight: 'bold' });

    rows.forEach((r, i) => {
        const y = 140 + i * rowH;
        // Faculty label on the left of everything
        s += label(60, y + 4, r[0], { size: 10, weight: 'bold' });

        // Remote contact
        s += `<circle cx="${leftRail}" cy="${y}" r="6" fill="${INK}"/>`;
        s += label(leftRail + 12, y + 4, r[1], { size: 8, color: INK });

        // Local contact (dashed = not built yet)
        s += `<circle cx="${rightRail}" cy="${y}" r="6" fill="none" stroke="${INK}" stroke-width="1.2" stroke-dasharray="2 2"/>`;
        s += label(rightRail + 12, y + 4, r[2], { size: 8, color: MID });

        // Changeover switch resting on remote
        const swX = leftRail + 120;
        s += `<line x1="${leftRail + 6}" y1="${y}" x2="${swX}" y2="${y}" stroke="${INK}" stroke-width="1.4"/>`;
        s += `<line x1="${swX}" y1="${y}" x2="${swX + 60}" y2="${y - 18}" stroke="${INK}" stroke-width="1.6"/>`;
        s += `<circle cx="${swX}" cy="${y}" r="3" fill="${INK}"/>`;
        s += `<circle cx="${swX + 60}" cy="${y - 18}" r="3" fill="${INK}"/>`;
        // Dashed alternate throw down to Pi rail
        s += `<line x1="${swX}" y1="${y}" x2="${rightRail - 6}" y2="${y}" stroke="${MID}" stroke-width="0.7" stroke-dasharray="4 3"/>`;

        // Flag line up from switch to a bus at the top
        s += `<line x1="${swX}" y1="${y}" x2="${swX}" y2="${busY}" stroke="${MID}" stroke-width="0.5" stroke-dasharray="2 3"/>`;
    });

    // Flag bus at the bottom feeding the prompt
    s += `<line x1="${leftRail + 60}" y1="${busY}" x2="${W - 200}" y2="${busY}" stroke="${INK}" stroke-width="1.4"/>`;
    s += `<polyline points="${W - 200},${busY} ${W - 150},${busY - 8} ${W - 100},${busY - 8}" stroke="${INK}" stroke-width="1.2" fill="none"/>`;
    s += label(W - 100, busY - 12, 'DEGRADED FLAG → PERSONA PROMPT', { size: 8, color: INK, weight: 'bold', spacing: 1 });
    s += label(leftRail + 60, busY + 14, 'the robot knows its own state and can say so', { size: 8, italic: true, color: MID });

    s += titleBlock('6', 'FALLBACK CHANGEOVER', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 07  ·  CONTROL PLANE SURFACE (replaces port map + interface table)
// ═══════════════════════════════════════════════════════════════════════════
function figControlSurface() {
    const W = 1000, H = 620;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    // Central circle: the ZERO process
    const cx = 500, cy = 320;
    s += `<circle cx="${cx}" cy="${cy}" r="120" fill="none" stroke="${INK}" stroke-width="2"/>`;
    s += `<circle cx="${cx}" cy="${cy}" r="90" fill="url(#hatch)"/>`;
    s += label(cx, cy - 6, 'ZERO PROCESS', { size: 12, anchor: 'middle', weight: 'bold' });
    s += label(cx, cy + 10, 'one brain, four states', { size: 8, anchor: 'middle', color: MID, italic: true });

    // Inbound endpoints on the LEFT
    const endpoints = [
        ['/health',        'liveness probe',    'GET'],
        ['/zero/status',   'state + degradation','GET'],
        ['/zero/say',      'speak a line',      'POST'],
        ['/zero/turn',     'push-to-talk audio','POST'],
        ['/zero/turn_text','typed input',       'POST'],
        ['/zero/control',  'end conversation',  'POST'],
    ];
    endpoints.forEach((e, i) => {
        const a = (i - (endpoints.length - 1) / 2) * 0.28 + Math.PI;
        const rx = cx + Math.cos(a) * 200;
        const ry = cy + Math.sin(a) * 160;
        s += `<line x1="${rx}" y1="${ry}" x2="${cx + Math.cos(a) * 120}" y2="${cy + Math.sin(a) * 120}"
                stroke="${INK}" stroke-width="1"/>`;
        s += `<rect x="${rx - 100}" y="${ry - 12}" width="100" height="24" fill="none" stroke="${INK}" stroke-width="0.8"/>`;
        s += label(rx - 96, ry + 3, e[0], { size: 8, weight: 'bold' });
        s += label(rx - 4, ry + 3, e[2], { size: 7, anchor: 'end', color: MID });
        s += label(rx - 96, ry + 22, e[1], { size: 7, color: MID, italic: true });
    });
    s += label(60, 100, 'INBOUND  ·  AF-1 APPLICATION', { size: 9, spacing: 2, weight: 'bold' });
    s += label(60, 114, 'HTTP :8090  ·  bound to 0.0.0.0', { size: 8, color: MID });

    // Outbound calls on the RIGHT
    const outbound = [
        ['/transcribe',         ':9000 whisper',    'audio → text'],
        ['/tts',                ':9100 orpheus',    'text → wav 24 kHz'],
        ['/facts + /analyze',   ':8000 vision',     'depth, scene, VLM'],
        ['/perceive/*',         ':8000 embedders',  'face, voice, obj'],
        ['/api/chat',           ':11435 → :11434',  'language model'],
        ['/search',             ':8080 searxng',    'web search'],
    ];
    outbound.forEach((e, i) => {
        const a = (i - (outbound.length - 1) / 2) * 0.28;
        const rx = cx + Math.cos(a) * 200;
        const ry = cy + Math.sin(a) * 160;
        s += `<line x1="${rx}" y1="${ry}" x2="${cx + Math.cos(a) * 120}" y2="${cy + Math.sin(a) * 120}"
                stroke="${INK}" stroke-width="1"/>`;
        s += `<rect x="${rx}" y="${ry - 12}" width="120" height="24" fill="none" stroke="${INK}" stroke-width="0.8"/>`;
        s += label(rx + 4, ry + 3, e[0], { size: 8, weight: 'bold' });
        s += label(rx + 116, ry + 3, e[1], { size: 7, anchor: 'end', color: MID });
        s += label(rx + 4, ry + 22, e[2], { size: 7, color: MID, italic: true });
    });
    s += label(W - 60, 100, 'OUTBOUND  ·  GPU NODE VIA TUNNEL', { size: 9, spacing: 2, weight: 'bold', anchor: 'end' });
    s += label(W - 60, 114, 'HTTP over SSH forwards to 127.0.0.1', { size: 8, color: MID, anchor: 'end' });

    s += titleBlock('7', 'CONTROL PLANE SURFACE', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 08  ·  PERSISTENT STORES (replaces the persistent-stores table)
// Drawn as an archive of chambered plates, each labelled with what lives inside.
// ═══════════════════════════════════════════════════════════════════════════
function figStores() {
    // Widened and reflowed into two columns so every card has real breathing
    // room for its notes and blob line. Notes are wrapped in code so text
    // never spills into an adjacent card.
    const W = 1500, H = 1200;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    const stores = [
        { name: 'zero_memory.sqlite',    key: 'memories',            blob: 'emb float32',   note: 'protected flag keeps the last-conversation row alive under prune' },
        { name: 'zero_identity.sqlite',  key: 'people + embeddings', blob: 'vec float32',   note: 'kind column lets face and voice share one table' },
        { name: 'zero_guests.sqlite',    key: 'guest_samples',       blob: 'vec float32',   note: 'negative ids by construction, capped per guest and total' },
        { name: 'zero_episodes.sqlite',  key: 'episodes',            blob: 'payload JSON',  note: 'WAL mode, PRAGMA user_version migrations' },
        { name: 'zero_curiosity.sqlite', key: 'questions',           blob: '(no blob)',     note: 'unique source_key so an observation cannot queue twice' },
        { name: 'zero_objects.sqlite',   key: 'objects',             blob: 'vec float32',   note: 'taught object name bound to its embedding' },
        { name: 'interactions.jsonl',    key: 'per-speaker turns',   blob: 'ndjson',        note: 'one record per speaker per session, written under a lock' },
        { name: 'voiceprint.npy',        key: 'enrolled owner',      blob: 'numpy float32', note: 'raw array on disk, read once at startup' },
    ];

    // Wrap helper for the note line so long text stays inside the card.
    function wrap(text, maxChars) {
        const words = text.split(' ');
        const lines = [];
        let cur = '';
        for (const w of words) {
            if ((cur + ' ' + w).trim().length > maxChars) {
                lines.push(cur.trim());
                cur = w;
            } else cur += ' ' + w;
        }
        if (cur.trim()) lines.push(cur.trim());
        return lines;
    }

    const perRow = 2;
    const cellW = 640, cellH = 200;
    const gapX = 40, gapY = 30;
    const totalW = perRow * cellW + (perRow - 1) * gapX;
    const marginX = (W - totalW) / 2;

    stores.forEach((st, i) => {
        const row = Math.floor(i / perRow);
        const col = i % perRow;
        const x = marginX + col * (cellW + gapX);
        const y = 150 + row * (cellH + gapY);

        // Card with a subtle shadow line to give depth without a fill.
        s += `<rect x="${x + 8}" y="${y + 8}" width="${cellW}" height="${cellH}" fill="none" stroke="${MID}" stroke-width="0.6"/>`;
        s += `<rect x="${x}" y="${y}" width="${cellW}" height="${cellH}" fill="none" stroke="${INK}" stroke-width="1.6"/>`;
        // Header rule
        s += `<line x1="${x}" y1="${y + 40}" x2="${x + cellW}" y2="${y + 40}" stroke="${INK}" stroke-width="1"/>`;
        // Name (larger)
        s += label(x + 20, y + 28, st.name, { size: 12, weight: 'bold' });
        // Disk glyph
        s += `<circle cx="${x + cellW - 26}" cy="${y + 22}" r="9" fill="none" stroke="${INK}" stroke-width="1.1"/>`;
        s += `<circle cx="${x + cellW - 26}" cy="${y + 22}" r="3" fill="${INK}"/>`;
        // Body: KEY + BLOB + NOTE in a labelled grid
        s += label(x + 20, y + 66, 'KEY', { size: 7, color: MID, spacing: 2, weight: 'bold' });
        s += label(x + 20, y + 88, st.key, { size: 10 });
        s += label(x + 20, y + 116, 'BLOB', { size: 7, color: MID, spacing: 2, weight: 'bold' });
        s += label(x + 20, y + 138, st.blob, { size: 10 });
        // Wrapped note across ~46 chars per line
        const noteLines = wrap(st.note, 46);
        s += label(x + 20, y + 172, 'NOTE', { size: 7, color: MID, spacing: 2, weight: 'bold' });
        noteLines.forEach((ln, k) => {
            s += label(x + 80, y + 172 + k * 14, ln, { size: 9, color: MID, italic: true });
        });
    });

    // Sign convention key in its own panel below the grid
    const kx = (W - 900) / 2, ky = H - 260;
    s += `<rect x="${kx}" y="${ky}" width="900" height="100" fill="none" stroke="${INK}" stroke-width="1"/>`;
    s += `<line x1="${kx}" y1="${ky + 26}" x2="${kx + 900}" y2="${ky + 26}" stroke="${INK}" stroke-width="0.8"/>`;
    s += label(kx + 20, ky + 18, 'PERSON_ID SIGN CONVENTION', { size: 10, spacing: 2, weight: 'bold' });

    s += `<circle cx="${kx + 30}" cy="${ky + 50}" r="6" fill="${INK}"/>`;
    s += label(kx + 48, ky + 55, 'positive  ·  enrolled person in identity registry', { size: 10 });

    s += `<circle cx="${kx + 30}" cy="${ky + 74}" r="6" fill="none" stroke="${INK}" stroke-width="1.6"/>`;
    s += label(kx + 48, ky + 79, 'negative  ·  provisional guest, clustered by voice', { size: 10 });

    s += `<circle cx="${kx + 480}" cy="${ky + 50}" r="6" fill="none" stroke="${INK}" stroke-width="0.8" stroke-dasharray="1.5 1.5"/>`;
    s += label(kx + 498, ky + 55, 'null  ·  anonymous turn (text mode, or unplaced speaker)', { size: 10 });

    s += titleBlock('8', 'PERSISTENT STORES', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 09  ·  ENDPOINT LADDER
// A silence axis with each rule drawn as a valve on the pipe.
// ═══════════════════════════════════════════════════════════════════════════
function figEndpoint() {
    const W = 1000, H = 500;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    const y = 240;
    const x0 = 120, x1 = W - 120;
    // Pipe
    s += `<line x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" stroke="${INK}" stroke-width="3"/>`;
    s += `<line x1="${x0}" y1="${y - 3}" x2="${x0}" y2="${y + 3}" stroke="${INK}" stroke-width="2"/>`;
    s += label(x0, y + 20, 'SPEECH ENDS  ·  t = 0', { size: 8, anchor: 'start', color: MID });
    s += label(x1, y + 20, 'TURN COMMITTED', { size: 8, anchor: 'end', color: MID });

    // Timeline positions (ms since speech ends), mapped onto the pipe
    const totalMs = 1200;
    const sc = (x1 - x0) / totalMs;

    const rules = [
        { ms: 180,  name: 'SPECULATIVE STT FIRES',   note: 'transcribe audio so far, off the hot path' },
        { ms: 450,  name: 'SILENCE THRESHOLD',       note: 'vad.silence_ms' },
        { ms: 900,  name: 'ADAPTIVE ENDPOINT',       note: 'doubled if utterance under 600 ms of speech' },
        { ms: 550,  name: 'SEMANTIC HOLD CHECK',     note: 'last word of speculative transcript', tri: true },
        { ms: 1150, name: 'HOLD WAIT LIMIT',         note: '+ semantic_hold_wait_ms bounded' },
    ];

    rules.forEach((r, i) => {
        const x = x0 + r.ms * sc;
        const above = i % 2 === 0;
        const ry = above ? y - 90 : y + 90;
        s += `<line x1="${x}" y1="${y - 12}" x2="${x}" y2="${y + 12}" stroke="${r.tri ? ACC : INK}" stroke-width="2"/>`;
        s += `<circle cx="${x}" cy="${y}" r="5" fill="${r.tri ? ACC : INK}"/>`;
        s += `<line x1="${x}" y1="${above ? y - 12 : y + 12}" x2="${x}" y2="${ry}" stroke="${MID}" stroke-width="0.7"/>`;
        s += label(x, ry + (above ? -4 : 14), r.name, { size: 9, anchor: 'middle', weight: 'bold', color: r.tri ? ACC : INK });
        s += label(x, ry + (above ? -18 : 28), `${r.ms} ms`, { size: 7, anchor: 'middle', color: MID });
        s += label(x, ry + (above ? -34 : 44), r.note, { size: 7, anchor: 'middle', color: MID, italic: true });
    });

    // Tri-state legend
    const ly = H - 130;
    s += label(60, ly, 'SEMANTIC HOLD IS TRI-STATE', { size: 9, spacing: 1, weight: 'bold', color: ACC });
    s += label(60, ly + 16, 'true  → hold one more silence window (transcript ended mid-thought)', { size: 8 });
    s += label(60, ly + 30, 'false → commit now (transcript ended in a sentence-final word)', { size: 8 });
    s += label(60, ly + 44, 'none  → transcript still in flight; wait bounded, do not race', { size: 8, color: ACC });

    s += titleBlock('9', 'ENDPOINT DECISION LADDER', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 10  ·  IDENTITY DECISION SURFACE
// Face cosine vs voice cosine, with the accept regions shaded and the fused
// line drawn as a real acceptance boundary.
// ═══════════════════════════════════════════════════════════════════════════
function figIdentity() {
    const W = 1000, H = 620;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    // Two panels: same-person vs different-person
    const drawPanel = (ox, oy, w, h, title, agree) => {
        // Axes
        s += `<line x1="${ox}" y1="${oy + h}" x2="${ox + w}" y2="${oy + h}" stroke="${INK}" stroke-width="1"/>`;
        s += `<line x1="${ox}" y1="${oy}" x2="${ox}" y2="${oy + h}" stroke="${INK}" stroke-width="1"/>`;
        // Grid
        for (let i = 0; i <= 10; i++) {
            const xx = ox + (i / 10) * w;
            const yy = oy + h - (i / 10) * h;
            s += `<line x1="${xx}" y1="${oy}" x2="${xx}" y2="${oy + h}" stroke="${FAINT}" stroke-width="0.4"/>`;
            s += `<line x1="${ox}" y1="${yy}" x2="${ox + w}" y2="${yy}" stroke="${FAINT}" stroke-width="0.4"/>`;
        }
        // Ticks
        [0, 0.5, 1.0].forEach(v => {
            const xx = ox + v * w;
            const yy = oy + h - v * h;
            s += `<line x1="${xx}" y1="${oy + h}" x2="${xx}" y2="${oy + h + 4}" stroke="${INK}" stroke-width="0.8"/>`;
            s += label(xx, oy + h + 16, v.toFixed(1), { size: 8, anchor: 'middle', color: MID });
            s += `<line x1="${ox - 4}" y1="${yy}" x2="${ox}" y2="${yy}" stroke="${INK}" stroke-width="0.8"/>`;
            s += label(ox - 8, yy + 3, v.toFixed(1), { size: 8, anchor: 'end', color: MID });
        });
        // Axis labels
        s += label(ox + w / 2, oy + h + 32, 'FACE COSINE', { size: 8, anchor: 'middle', spacing: 2, weight: 'bold' });
        s += label(ox - 40, oy + h / 2, 'VOICE COSINE', { size: 8, anchor: 'middle', spacing: 2, weight: 'bold', color: INK });
        s += `<g transform="translate(${ox - 34},${oy + h / 2}) rotate(-90)"><text font-size="8" text-anchor="middle" font-family="Courier New, monospace" letter-spacing="2" font-weight="bold">VOICE COSINE</text></g>`;

        // Regions
        if (agree) {
            // Fused accept region: w_face * f + w_voice * v >= 0.5
            // w_face = 0.55, w_voice = 0.45
            const wf = 0.55, wv = 0.45, thr = 0.5;
            // f = (thr - wv * v) / wf
            const pts = [];
            for (let v = 0; v <= 1.001; v += 0.05) {
                const f = (thr - wv * v) / wf;
                if (f >= 0 && f <= 1) pts.push([ox + f * w, oy + h - v * h]);
            }
            // Close the polygon along top-right
            const region = pts.map(p => `${p[0]},${p[1]}`).join(' ') +
                ` ${ox + w},${oy} ${ox + w},${oy + h - 1 * h} `;
            s += `<polygon points="${region}" fill="url(#hatch)" opacity="0.9"/>`;
            s += `<polyline points="${pts.map(p => p.join(',')).join(' ')}" fill="none" stroke="${INK}" stroke-width="1.6"/>`;
            s += label(ox + w * 0.72, oy + h * 0.32, 'ACCEPT (fused)', { size: 9, anchor: 'middle', weight: 'bold' });
            s += label(ox + w * 0.72, oy + h * 0.44, 'fusion.threshold = 0.50', { size: 7, anchor: 'middle', color: MID });
        } else {
            // Disagreement: single-channel bands. face >= 0.42, voice >= 0.45
            const fThr = 0.42, vThr = 0.45;
            const fx = ox + fThr * w;
            const vy = oy + h - vThr * h;
            s += `<rect x="${fx}" y="${oy}" width="${w - fThr * w}" height="${h}" fill="url(#hatch)" opacity="0.6"/>`;
            s += `<rect x="${ox}" y="${oy}" width="${w}" height="${vy - oy}" fill="url(#hatch)" opacity="0.6"/>`;
            s += `<line x1="${fx}" y1="${oy}" x2="${fx}" y2="${oy + h}" stroke="${INK}" stroke-width="1.4"/>`;
            s += `<line x1="${ox}" y1="${vy}" x2="${ox + w}" y2="${vy}" stroke="${INK}" stroke-width="1.4"/>`;
            s += label(fx + 4, oy + 14, 'face_threshold 0.42', { size: 7, color: INK });
            s += label(ox + 4, vy - 4, 'voice_threshold 0.45', { size: 7, color: INK });
            s += label(ox + w * 0.65, oy + h * 0.85, 'only the stronger single', { size: 9, anchor: 'middle', italic: true });
            s += label(ox + w * 0.65, oy + h * 0.92, 'channel competes; both must clear', { size: 8, anchor: 'middle', italic: true, color: MID });
        }
        // Title
        s += label(ox + w / 2, oy - 12, title, { size: 9, anchor: 'middle', spacing: 2, weight: 'bold' });
    };

    drawPanel(120, 100, 340, 340, 'PANEL A  ·  BOTH CHANNELS AGREE ON A NAME', true);
    drawPanel(560, 100, 340, 340, 'PANEL B  ·  CHANNELS NAME DIFFERENT PEOPLE', false);

    s += label(W / 2, H - 130, 'THE TWO PANELS ARE WHY WEAK AGREEING SIGNALS ACCEPT AND WEAK CONFLICTING ONES DO NOT',
        { size: 8, anchor: 'middle', spacing: 2, color: MID });

    s += titleBlock('10', 'IDENTITY DECISION SURFACE', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 11  ·  MEMORY ACTIVATION DECAY
// ═══════════════════════════════════════════════════════════════════════════
function figDecay() {
    const W = 1000, H = 520;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    const ox = 130, oy = 90, w = W - 260, h = 320;

    // Axes
    s += `<line x1="${ox}" y1="${oy + h}" x2="${ox + w}" y2="${oy + h}" stroke="${INK}" stroke-width="1"/>`;
    s += `<line x1="${ox}" y1="${oy}" x2="${ox}" y2="${oy + h}" stroke="${INK}" stroke-width="1"/>`;

    // Grid + ticks
    for (let d = 0; d <= 60; d += 5) {
        const xx = ox + (d / 60) * w;
        const major = d % 10 === 0;
        s += `<line x1="${xx}" y1="${oy + h}" x2="${xx}" y2="${oy + h + (major ? 6 : 3)}" stroke="${INK}" stroke-width="${major ? 0.9 : 0.5}"/>`;
        if (major) s += label(xx, oy + h + 18, `${d}d`, { size: 8, anchor: 'middle', color: MID });
    }
    for (let v = 0; v <= 1.001; v += 0.1) {
        const yy = oy + h - v * h;
        const major = Math.round(v * 10) % 5 === 0;
        s += `<line x1="${ox - (major ? 6 : 3)}" y1="${yy}" x2="${ox}" y2="${yy}" stroke="${INK}" stroke-width="${major ? 0.9 : 0.5}"/>`;
        if (major) s += label(ox - 10, yy + 3, v.toFixed(1), { size: 8, anchor: 'end', color: MID });
        s += `<line x1="${ox}" y1="${yy}" x2="${ox + w}" y2="${yy}" stroke="${FAINT}" stroke-width="0.4"/>`;
    }

    // Curve: 0.2 + 0.8 * exp(-age / 14 days), scaled to 60 days
    const tau = 14;
    const pts = [];
    for (let d = 0; d <= 60; d += 0.5) {
        const y = 0.2 + 0.8 * Math.exp(-d / tau);
        pts.push([ox + (d / 60) * w, oy + h - y * h]);
    }
    s += `<polyline points="${pts.map(p => p.join(',')).join(' ')}" fill="none" stroke="${INK}" stroke-width="2"/>`;

    // 0.2 floor
    const fy = oy + h - 0.2 * h;
    s += `<line x1="${ox}" y1="${fy}" x2="${ox + w}" y2="${fy}" stroke="${MID}" stroke-width="0.9" stroke-dasharray="4 3"/>`;
    s += label(ox + w * 0.7, fy - 6, 'FLOOR 0.2  ·  keeps old-but-important memories reachable', { size: 8, color: MID });

    // Configured "half-life" mark (14 days) — really a time constant
    const cx = ox + (14 / 60) * w;
    s += `<line x1="${cx}" y1="${oy}" x2="${cx}" y2="${oy + h}" stroke="${ACC}" stroke-width="0.9" stroke-dasharray="3 3"/>`;
    s += label(cx + 6, oy + 18, 'CONFIGURED "half_life_days" = 14', { size: 8, color: ACC });
    s += label(cx + 6, oy + 32, '(behaves as a time constant, not a half-life)', { size: 7, color: ACC, italic: true });

    // True half-life (~9.7 days)
    const hx = ox + (9.7 / 60) * w;
    s += `<line x1="${hx}" y1="${oy}" x2="${hx}" y2="${oy + h}" stroke="${INK}" stroke-width="0.9" stroke-dasharray="1 3"/>`;
    s += label(hx - 8, oy + 18, 'TRUE HALF-LIFE ≈ 9.7 d', { size: 8, anchor: 'end', color: INK });

    // Axis labels
    s += label(ox + w / 2, oy + h + 40, 'AGE SINCE LAST ACCESS (days)', { size: 8, anchor: 'middle', spacing: 2, weight: 'bold' });
    s += `<g transform="translate(${ox - 44},${oy + h / 2}) rotate(-90)"><text font-size="8" text-anchor="middle" font-family="Courier New, monospace" letter-spacing="2" font-weight="bold">RECENCY CONTRIBUTION</text></g>`;

    s += titleBlock('11', 'MEMORY ACTIVATION DECAY', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 12  ·  BUILD & DEPLOY  ·  Exploded assembly
// ═══════════════════════════════════════════════════════════════════════════
function figBuildAssembly() {
    const W = 1000, H = 660;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    // Two columns: Pi head stack (left) and GPU node stack (right)
    const drawStack = (ox, layers, title) => {
        s += label(ox + 120, 90, title, { size: 10, anchor: 'middle', spacing: 2, weight: 'bold' });
        layers.forEach((l, i) => {
            const y = 130 + i * 66;
            // Plate
            s += `<rect x="${ox}" y="${y}" width="240" height="48" fill="none" stroke="${INK}" stroke-width="1.4"/>`;
            // Bolts
            s += `<circle cx="${ox + 8}" cy="${y + 8}" r="2.5" fill="${INK}"/>`;
            s += `<circle cx="${ox + 232}" cy="${y + 8}" r="2.5" fill="${INK}"/>`;
            s += `<circle cx="${ox + 8}" cy="${y + 40}" r="2.5" fill="${INK}"/>`;
            s += `<circle cx="${ox + 232}" cy="${y + 40}" r="2.5" fill="${INK}"/>`;
            // Layer label
            s += label(ox + 14, y + 20, l.name, { size: 9, weight: 'bold' });
            s += label(ox + 14, y + 38, l.note, { size: 7, color: MID });
            // Explosion dashes between layers
            if (i < layers.length - 1) {
                const dy = y + 48;
                for (let k = 0; k < 4; k++) {
                    s += `<line x1="${ox + 120}" y1="${dy + k * 4}" x2="${ox + 120}" y2="${dy + k * 4 + 2}" stroke="${MID}" stroke-width="0.6"/>`;
                }
            }
        });
    };
    drawStack(80, [
        { name: 'Raspberry Pi OS Lite',  note: 'bookworm, 64-bit' },
        { name: 'PortAudio + PipeWire',  note: 'audio device stack' },
        { name: 'Python venv + wheels',  note: 'zero package + optional vision' },
        { name: 'ONNX Runtime',          note: 'wake, VAD, YOLO, ArcFace, ECAPA' },
        { name: 'Pi Ollama (opt.)',      note: 'local LLM fallback on :11434' },
        { name: 'systemd units',         note: 'zero, zero-tunnel, zero-preview' },
    ], 'PI 5 (HEAD)');

    drawStack(680, [
        { name: 'CUDA + drivers',        note: 'card visible to torch and vLLM' },
        { name: 'Faster-Whisper server', note: 'large-v3-turbo on :9000' },
        { name: 'Orpheus vLLM server',   note: '3B speech model on :9100' },
        { name: 'Vision FastAPI',        note: 'depth + VLM + embedders on :8000' },
        { name: 'GPU Ollama',            note: 'gemma4 + nomic-embed on :11434' },
        { name: 'SearXNG (opt.)',        note: 'private web search on :8080' },
    ], 'GPU NODE');

    // Center: tunnel binding both stacks together
    const midX = W / 2;
    s += `<line x1="${midX}" y1="140" x2="${midX}" y2="530" stroke="${INK}" stroke-width="1.6"/>`;
    for (let y = 150; y <= 520; y += 10) {
        s += `<line x1="${midX - 10}" y1="${y}" x2="${midX + 10}" y2="${y}" stroke="${INK}" stroke-width="0.6"/>`;
    }
    s += label(midX, 128, 'SSH TUNNEL', { size: 9, anchor: 'middle', weight: 'bold', spacing: 2 });
    s += label(midX, 546, 'autossh, five forwards', { size: 8, anchor: 'middle', color: MID });

    s += titleBlock('12', 'BUILD ASSEMBLY  ·  EXPLODED VIEW', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 13  ·  TEST RIG (a scope, a mic, a stopwatch)
// ═══════════════════════════════════════════════════════════════════════════
function figTestRig() {
    const W = 1000, H = 540;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    // Central rig table
    s += `<line x1="80" y1="380" x2="${W - 80}" y2="380" stroke="${INK}" stroke-width="1.6"/>`;
    s += `<line x1="120" y1="380" x2="120" y2="430" stroke="${INK}" stroke-width="1.2"/>`;
    s += `<line x1="${W - 120}" y1="380" x2="${W - 120}" y2="430" stroke="${INK}" stroke-width="1.2"/>`;

    // Oscilloscope box
    s += `<rect x="150" y="130" width="260" height="180" fill="none" stroke="${INK}" stroke-width="1.6"/>`;
    s += `<rect x="170" y="150" width="220" height="120" fill="none" stroke="${INK}" stroke-width="0.9"/>`;
    // Grid inside scope
    for (let i = 1; i < 10; i++) s += `<line x1="${170 + i * 22}" y1="150" x2="${170 + i * 22}" y2="270" stroke="${FAINT}" stroke-width="0.4"/>`;
    for (let i = 1; i < 6; i++) s += `<line x1="170" y1="${150 + i * 20}" x2="390" y2="${150 + i * 20}" stroke="${FAINT}" stroke-width="0.4"/>`;
    // A latency waveform
    const pts = [];
    for (let x = 0; x <= 220; x++) {
        const t = x / 220;
        const y = 210 + Math.sin(t * Math.PI * 4) * 20 * (1 - t) - (t > 0.6 ? 40 : 0) * Math.exp(-(t - 0.6) * 6);
        pts.push([170 + x, y]);
    }
    s += `<polyline points="${pts.map(p => p.join(',')).join(' ')}" fill="none" stroke="${INK}" stroke-width="1.2"/>`;
    s += label(280, 296, 'FIRST-AUDIO LATENCY TRACE', { size: 7, anchor: 'middle', color: MID, spacing: 1 });
    // Knobs
    [40, 90, 140].forEach((cy, i) => {
        s += `<circle cx="${420}" cy="${170 + i * 50}" r="12" fill="none" stroke="${INK}" stroke-width="1"/>`;
        s += `<line x1="${420}" y1="${170 + i * 50}" x2="${420 + 8}" y2="${170 + i * 50 - 6}" stroke="${INK}" stroke-width="1.2"/>`;
    });
    s += label(280, 122, 'SCOPE  ·  turn timing capture', { size: 8, anchor: 'middle', spacing: 2, weight: 'bold' });

    // Microphone + person
    s += `<circle cx="600" cy="200" r="34" fill="none" stroke="${INK}" stroke-width="1.6"/>`;
    for (let i = 0; i < 6; i++) s += `<line x1="${600 - 20 + i * 8}" y1="180" x2="${600 - 20 + i * 8}" y2="220" stroke="${INK}" stroke-width="0.8"/>`;
    s += `<line x1="600" y1="234" x2="600" y2="380" stroke="${INK}" stroke-width="1.4"/>`;
    s += `<circle cx="600" cy="380" r="10" fill="none" stroke="${INK}" stroke-width="1.2"/>`;
    s += label(600, 260, 'REFERENCE UTTERANCE', { size: 8, anchor: 'middle', spacing: 2, weight: 'bold' });
    s += label(600, 274, 'known text, known duration', { size: 7, anchor: 'middle', color: MID, italic: true });

    // Stopwatch
    s += `<circle cx="780" cy="200" r="40" fill="none" stroke="${INK}" stroke-width="1.6"/>`;
    s += `<circle cx="780" cy="200" r="32" fill="none" stroke="${INK}" stroke-width="0.8"/>`;
    for (let i = 0; i < 12; i++) {
        const a = i * Math.PI / 6 - Math.PI / 2;
        s += `<line x1="${780 + Math.cos(a) * 26}" y1="${200 + Math.sin(a) * 26}" x2="${780 + Math.cos(a) * 30}" y2="${200 + Math.sin(a) * 30}" stroke="${INK}" stroke-width="1"/>`;
    }
    s += `<line x1="780" y1="200" x2="${780 + Math.cos(-Math.PI / 3) * 22}" y2="${200 + Math.sin(-Math.PI / 3) * 22}" stroke="${ACC}" stroke-width="2"/>`;
    s += `<line x1="780" y1="200" x2="${780 + Math.cos(Math.PI / 5) * 14}" y2="${200 + Math.sin(Math.PI / 5) * 14}" stroke="${INK}" stroke-width="1.6"/>`;
    s += `<rect x="770" y="152" width="20" height="8" fill="none" stroke="${INK}" stroke-width="1.2"/>`;
    s += label(780, 260, 'END-TO-END WATCH', { size: 8, anchor: 'middle', spacing: 2, weight: 'bold' });
    s += label(780, 274, 'wake ➜ first audio', { size: 7, anchor: 'middle', color: MID, italic: true });

    // Signal path
    s += `<path d="M 634 200 C 700 200, 720 200, 760 200" stroke="${INK}" stroke-width="0.8" stroke-dasharray="3 2" fill="none"/>`;
    s += `<path d="M 566 200 C 500 200, 440 200, 410 200" stroke="${INK}" stroke-width="0.8" stroke-dasharray="3 2" fill="none"/>`;

    s += titleBlock('13', 'TEST RIG', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 14  ·  FAILURE TRIAGE (as a circuit-breaker panel)
// ═══════════════════════════════════════════════════════════════════════════
function figFailureBreakers() {
    const W = 1000, H = 620;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    // Rail
    s += `<line x1="80" y1="120" x2="80" y2="${H - 130}" stroke="${INK}" stroke-width="2"/>`;
    s += label(80, 106, 'RAIL', { size: 8, anchor: 'middle', spacing: 2, weight: 'bold' });

    const breakers = [
        { name: 'TUNNEL LOSS',           trip: 'client raise', resp: 'lazy local STT/TTS, degraded flag, robot narrates it' },
        { name: 'CAMERA ABSENT',         trip: 'None from factory', resp: 'voice-only path, LLM told "you are blind"' },
        { name: 'GPU SERVER SLOW',       trip: 'embed timeout 3 s', resp: 'auto-degrade to hash embeddings for the session' },
        { name: 'OLLAMA COLD/OOM',       trip: 'stream error', resp: 'retry once, then say so; KV quantised to keep resident' },
        { name: 'AUDIO DEVICE GONE',     trip: 'PortAudio raise', resp: 'process refuses to start; systemd Restart=on-failure' },
        { name: 'WHISPER HALLUCINATES',  trip: 'gate: min words/ms/rms', resp: 'no phantom guest minted, corpus stays clean' },
        { name: 'MEMORY DB LOCK',        trip: 'sqlite busy', resp: 'skip write, retry next turn; WAL for readers' },
        { name: 'CONTROL EXCEPTION',     trip: 'handler catch', resp: 'HTTP 500 with 200-char excerpt; process survives' },
    ];

    breakers.forEach((b, i) => {
        const y = 130 + i * 52;
        // Bus tap
        s += `<line x1="80" y1="${y}" x2="120" y2="${y}" stroke="${INK}" stroke-width="1.4"/>`;
        // Breaker body
        s += `<rect x="120" y="${y - 20}" width="80" height="40" fill="none" stroke="${INK}" stroke-width="1.4"/>`;
        // Toggle
        s += `<line x1="140" y1="${y}" x2="180" y2="${y - 10}" stroke="${INK}" stroke-width="1.8"/>`;
        s += `<circle cx="140" cy="${y}" r="3" fill="${INK}"/>`;
        s += `<circle cx="180" cy="${y - 10}" r="3" fill="${INK}"/>`;
        // Name
        s += label(210, y - 5, b.name, { size: 9, weight: 'bold' });
        // Trip label
        s += label(210, y + 8, `trip: ${b.trip}`, { size: 7, color: MID, italic: true });
        // Response
        s += label(500, y + 3, b.resp, { size: 8, color: INK });
    });

    s += titleBlock('14', 'FAILURE TRIAGE  ·  BREAKER PANEL', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 15  ·  CONFIG ANATOMY (tree of top-level blocks)
// ═══════════════════════════════════════════════════════════════════════════
function figConfigTree() {
    // Wider canvas, fewer columns (5) so each node has room for its hint
    // line without colliding with the next node.
    const W = 1600, H = 1200;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    // Root
    const root = { x: W / 2, y: 140 };
    s += `<rect x="${root.x - 140}" y="${root.y - 34}" width="280" height="68" fill="url(#hatch)" stroke="${INK}" stroke-width="1.8"/>`;
    s += label(root.x, root.y - 6, 'config.yaml', { size: 16, anchor: 'middle', weight: 'bold' });
    s += label(root.x, root.y + 18, 'deep-merged with config.local.yaml at load', { size: 9, anchor: 'middle', color: MID, italic: true });

    const blocks = [
        { n: 'audio',        h: 'sample rate, gain, aec' },
        { n: 'privacy',      h: 'bystander_mode' },
        { n: 'control',      h: 'host, port, person_id' },
        { n: 'conversation', h: 'sleep, filler, barge in' },
        { n: 'memory',       h: 'retrieval, forgetting' },
        { n: 'learning',     h: 'objects, corpus, episodes' },
        { n: 'perception',   h: 'remote, affect, diarize' },
        { n: 'proactive',    h: 'quiet_hours, cooldowns' },
        { n: 'preferences',  h: 'enabled' },
        { n: 'tools',        h: 'allow, websearch' },
        { n: 'identity',     h: 'fusion, guests, session' },
        { n: 'voiceid',      h: 'enabled, threshold' },
        { n: 'wake',         h: 'model, threshold' },
        { n: 'vad',          h: 'silence, semantic_hold' },
        { n: 'stt',          h: 'engine, remote, fallback' },
        { n: 'llm',          h: 'host, model, history' },
        { n: 'tts',          h: 'engine, orpheus, piper' },
        { n: 'vision',       h: 'camera, detect, gpu' },
        { n: 'world',        h: 'motion, gate, surprise' },
    ];
    const cols = 5;
    const rows = Math.ceil(blocks.length / cols);
    const cellW = 260, cellH = 130;
    const marginX = (W - cols * cellW) / 2;

    blocks.forEach((b, i) => {
        const col = i % cols;
        const row = Math.floor(i / cols);
        const x = marginX + col * cellW + cellW / 2;
        const y = 340 + row * cellH;

        // Node
        s += `<rect x="${x - 100}" y="${y - 30}" width="200" height="60" fill="none" stroke="${INK}" stroke-width="1.4"/>`;
        s += label(x, y - 2, b.n, { size: 12, anchor: 'middle', weight: 'bold' });
        s += label(x, y + 18, b.h, { size: 8, anchor: 'middle', color: MID, italic: true });

        // Curved line up to the root (short, non intersecting)
        s += `<path d="M ${x} ${y - 30} C ${x} ${root.y + 100}, ${root.x} ${root.y + 100}, ${root.x} ${root.y + 34}"
                fill="none" stroke="${MID}" stroke-width="0.5"/>`;
    });

    s += titleBlock('15', 'CONFIGURATION ANATOMY', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 16  ·  LATENCY WATERFALL + POWER
// ═══════════════════════════════════════════════════════════════════════════
function figLatency() {
    const W = 1000, H = 600;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    // Waterfall
    const stages = [
        ['WAKE ACCEPT',      0,   40],
        ['ENDPOINT COMMIT',  0,   450],
        ['STT (spec re-use)',450, 60],
        ['RECALL',           510, 280],
        ['LLM PREFILL',      510, 380],
        ['FILLER PLAYED',    790, 350],
        ['LLM FIRST TOKEN',  890, 60],
        ['TTS FIRST CHUNK',  950, 220],
        ['FIRST AUDIO OUT', 1170, 20],
    ];

    const x0 = 220, x1 = W - 220;
    const tMax = 1400;
    const scale = (x1 - x0) / tMax;
    // Axis
    s += `<line x1="${x0}" y1="120" x2="${x0}" y2="${H - 200}" stroke="${INK}" stroke-width="1"/>`;
    for (let t = 0; t <= tMax; t += 200) {
        const y = 120 + (t / tMax) * (H - 320);
        s += `<line x1="${x0 - 4}" y1="${y}" x2="${x1 + 4}" y2="${y}" stroke="${FAINT}" stroke-width="0.4"/>`;
        s += label(x0 - 8, y + 3, `${t} ms`, { size: 8, anchor: 'end', color: MID });
    }

    stages.forEach((st, i) => {
        const [name, start, dur] = st;
        const y0 = 120 + (start / tMax) * (H - 320);
        const y1 = 120 + ((start + dur) / tMax) * (H - 320);
        const x = x0 + 30 + i * 55;
        // Bar
        const color = name === 'FIRST AUDIO OUT' ? ACC : INK;
        s += `<line x1="${x}" y1="${y0}" x2="${x}" y2="${y1}" stroke="${color}" stroke-width="6"/>`;
        s += label(x, y1 + 18, name, { size: 7, anchor: 'middle', color });
    });

    // Power meter on the right
    const px = W - 160;
    s += `<rect x="${px}" y="140" width="90" height="260" fill="none" stroke="${INK}" stroke-width="1.4"/>`;
    // Ticks
    for (let w = 0; w <= 20; w += 5) {
        const y = 400 - (w / 20) * 260;
        s += `<line x1="${px - 4}" y1="${y}" x2="${px + 4}" y2="${y}" stroke="${INK}" stroke-width="0.8"/>`;
        s += label(px - 8, y + 3, `${w} W`, { size: 8, anchor: 'end', color: MID });
    }
    // Idle band
    s += `<rect x="${px}" y="${400 - (3 / 20) * 260}" width="90" height="${(3 / 20) * 260}" fill="url(#hatch)"/>`;
    // Peak
    s += `<line x1="${px}" y1="${400 - (12 / 20) * 260}" x2="${px + 90}" y2="${400 - (12 / 20) * 260}" stroke="${ACC}" stroke-width="1.6"/>`;
    s += label(px + 45, 128, 'PI 5 POWER', { size: 8, anchor: 'middle', spacing: 2, weight: 'bold' });
    s += label(px + 95, 400 - (3 / 20) * 260 + 4, 'idle band', { size: 7, color: MID });
    s += label(px + 95, 400 - (12 / 20) * 260 + 4, 'peak during THINK+SPEAK', { size: 7, color: ACC });

    s += titleBlock('16', 'LATENCY WATERFALL AND POWER', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 17  ·  PRIVACY LAYERS (a vault cross-section)
// ═══════════════════════════════════════════════════════════════════════════
function figPrivacy() {
    const W = 1000, H = 580;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    // Vault: three concentric rings
    const cx = 320, cy = 300;
    const rings = [
        { r: 200, name: 'OPEN', note: 'answered and remembered · single-user household' },
        { r: 140, name: 'GUARDED', note: 'answered, nothing stored long-term · default' },
        { r: 80,  name: 'STRICT',  note: 'unknown voices not engaged at all' },
    ];
    rings.forEach((rg, i) => {
        s += `<circle cx="${cx}" cy="${cy}" r="${rg.r}" fill="none" stroke="${INK}" stroke-width="${1.6 - i * 0.2}"/>`;
        s += label(cx, cy - rg.r + 16, rg.name, { size: 9, anchor: 'middle', weight: 'bold', spacing: 2 });
        s += label(cx, cy - rg.r + 30, rg.note, { size: 7, anchor: 'middle', color: MID, italic: true });
    });
    // Enrolled always at the middle
    s += `<circle cx="${cx}" cy="${cy}" r="26" fill="url(#hatch)" stroke="${INK}" stroke-width="1.4"/>`;
    s += label(cx, cy + 3, 'ENROLLED', { size: 8, anchor: 'middle', weight: 'bold' });
    s += label(cx, cy + 14, 'always full service', { size: 7, anchor: 'middle', color: MID, italic: true });

    // On the right: the visible indicator + erasure verbs
    s += label(700, 130, 'ALWAYS ON', { size: 9, spacing: 2, weight: 'bold' });
    s += `<circle cx="720" cy="170" r="12" fill="${ACC}" stroke="${INK}" stroke-width="1.4"/>`;
    s += label(742, 174, 'GPIO INDICATOR LED', { size: 9, weight: 'bold' });
    s += label(742, 188, 'lit for listening / thinking / speaking', { size: 7, color: MID, italic: true });

    s += label(700, 240, 'ALWAYS HONOURED', { size: 9, spacing: 2, weight: 'bold' });
    ['"forget that"', '"forget everything about me"', '"stop remembering"'].forEach((v, i) => {
        s += label(720, 265 + i * 18, v, { size: 9 });
    });

    s += label(700, 360, 'HELD BY THE SUBSYSTEM', { size: 9, spacing: 2, weight: 'bold' });
    ['face embedding (ArcFace 512-d)', 'voice embedding (ECAPA 512-d)',
     'transcripts (per-speaker JSONL)', 'derived facts + preferences (SQLite)'].forEach((v, i) => {
        s += `<circle cx="710" cy="${380 + i * 18}" r="2.5" fill="${INK}"/>`;
        s += label(722, 384 + i * 18, v, { size: 9 });
    });

    s += titleBlock('17', 'PRIVACY LAYERS', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 18  ·  OBSERVABILITY (log flow)
// ═══════════════════════════════════════════════════════════════════════════
function figObservability() {
    const W = 1000, H = 560;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    // A meandering log stream with tagged blocks
    const path = 'M 90 120 C 260 100, 340 200, 500 180 S 780 300, 920 260 L 920 340 C 780 380, 500 360, 340 400 S 160 480, 90 460';
    s += `<path d="${path}" fill="none" stroke="${INK}" stroke-width="2"/>`;

    const marks = [
        { x: 150, y: 118, name: 'wake:', text: 'accepted score 0.72' },
        { x: 320, y: 170, name: 'vad:', text: 'utterance rms 3120 peak 12480' },
        { x: 500, y: 180, name: 'stt-spec:', text: '"what is the" (partial)' },
        { x: 680, y: 220, name: 'endpoint:', text: 'committed after 450 ms' },
        { x: 850, y: 260, name: 'llm:', text: 'first token in 620 ms' },
        { x: 800, y: 340, name: 'tts:', text: 'first audio in 220 ms' },
        { x: 480, y: 380, name: 'affect:', text: 'valence +0.3 confident' },
        { x: 260, y: 430, name: 'memory:', text: 'wrote 2 facts under person 3' },
    ];
    marks.forEach(m => {
        s += `<circle cx="${m.x}" cy="${m.y}" r="4" fill="${INK}"/>`;
        s += label(m.x + 8, m.y - 4, m.name, { size: 8, weight: 'bold' });
        s += label(m.x + 8, m.y + 8, m.text, { size: 7, color: MID });
    });

    // Load-bearing lines callout
    s += label(60, H - 130, 'LOAD-BEARING TRIAGE LINES  ·  do not remove:',
        { size: 8, weight: 'bold', spacing: 2 });
    ['listening… heartbeat every 5 s', 'utterance: rms=… peak=…',
     'barge-in armed: … speech gate', 'degraded: STT/TTS switched to local'].forEach((v, i) => {
        s += label(70, H - 110 + i * 14, `· ${v}`, { size: 8 });
    });

    s += titleBlock('18', 'A SESSION READ FROM THE LOGS', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 19  ·  RUNBOOK FLOW (start/stop/triage)
// ═══════════════════════════════════════════════════════════════════════════
function figRunbook() {
    // Widened, all boxes bigger with generous line height so nothing overlaps.
    const W = 1600, H = 1200;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    const box = (x, y, w, h, title, lines) => {
        s += `<rect x="${x}" y="${y}" width="${w}" height="${h}" fill="none" stroke="${INK}" stroke-width="1.6"/>`;
        s += `<line x1="${x}" y1="${y + 42}" x2="${x + w}" y2="${y + 42}" stroke="${INK}" stroke-width="0.9"/>`;
        s += label(x + 14, y + 28, title, { size: 11, weight: 'bold', spacing: 2 });
        lines.forEach((l, i) => s += label(x + 14, y + 70 + i * 24, l, { size: 9 }));
    };
    const arrow = (x1, y1, x2, y2) => {
        s += `<path d="M ${x1} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}"
                fill="none" stroke="${INK}" stroke-width="1.2"/>`;
        s += `<rect x="${x2 - 4}" y="${y2 - 4}" width="8" height="8" fill="${INK}"/>`;
    };

    const boxW = 440, boxH = 220, gap = 40;
    // Column 1: bring up steps
    box(80,  150, boxW, boxH, 'ON THE GPU NODE', [
        '1.  bash scripts/run_gpu_servers.sh',
        '2.  curl :9000/health, :9100/health, :8000/health',
        '3.  curl :11434/api/tags   (list of models)',
    ]);
    box(80,  150 + boxH + gap, boxW, boxH, 'ON THE PI  (TUNNEL)', [
        '1.  GPU_HOST=... bash scripts/pi_tunnel.sh',
        '2.  curl 127.0.0.1:9000/health',
        '3.  curl 127.0.0.1:11435/api/tags',
    ]);
    box(80,  150 + 2 * (boxH + gap), boxW, boxH, 'ON THE PI  (ZERO)', [
        '1.  systemctl start zero',
        '2.  poll 127.0.0.1:8090/health for ready:true',
        '3.  say the wake word',
    ]);

    // Column 2: triage boxes
    box(80 + boxW + gap, 150, boxW, boxH, 'TRIAGE  ·  NO REPLY', [
        'check vad "listening..." heartbeat',
        'if no wake acceptance: raise or lower wake.threshold',
        'if no endpoint commit: check tunnel to :9000',
    ]);
    box(80 + boxW + gap, 150 + boxH + gap, boxW, boxH, 'TRIAGE  ·  ROBOT SLOW', [
        'read degraded flag in /zero/status',
        'ollama ps  (chat model resident?)',
        'grep for slow embed lines in the log',
    ]);
    box(80 + boxW + gap, 150 + 2 * (boxH + gap), boxW, boxH, 'TRIAGE  ·  DEAF AFTER REPLY', [
        'check monitor thread joined before mic pause',
        'confirm producer sentinel blocking put in use',
        'these were the two barge-in shutdown bugs',
    ]);

    // Column 3: deploy and backup
    box(80 + 2 * (boxW + gap), 150, boxW, boxH, 'DEPLOY A CONFIG CHANGE', [
        '1.  edit config.local.yaml',
        '2.  systemctl restart zero',
        '3.  journalctl -u zero -f',
        '4.  curl /health, wait for ready:true',
    ]);
    box(80 + 2 * (boxW + gap), 150 + boxH + gap, boxW, boxH, 'BACKUP', [
        'stop zero, cp the six sqlite files,',
        'cp voiceprint.npy and data/corpus/,',
        'start zero.',
    ]);
    box(80 + 2 * (boxW + gap), 150 + 2 * (boxH + gap), boxW, boxH, 'RESTORE', [
        'stop zero, replace the files in place,',
        'start zero,',
        'verify by asking the robot to recall a known fact.',
    ]);

    // Arrows across columns to show the flow
    arrow(80 + boxW,        260, 80 + boxW + gap,       260);
    arrow(80 + boxW,        260 + boxH + gap, 80 + boxW + gap,       260 + boxH + gap);
    arrow(80 + boxW,        260 + 2 * (boxH + gap), 80 + boxW + gap, 260 + 2 * (boxH + gap));

    s += titleBlock('19', 'OPERATIONS RUNBOOK', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 20  ·  DEPENDENCY STACK (columns of third-party parts)
// ═══════════════════════════════════════════════════════════════════════════
function figDependencies() {
    // Widened so the LICENCES column no longer clips at the right border.
    const W = 1500, H = 900;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    const columns = [
        {
            title: 'CODE',
            items: [
                'Python 3.11+',
                'onnxruntime (CPU build)',
                'numpy, opencv-python',
                'sounddevice / pyaudio',
                'python-webrtcvad',
                'gpiozero (Pi only)',
                'FastAPI (server side)',
                'vLLM (server side)',
                'ollama-python',
            ],
        },
        {
            title: 'MODELS',
            items: [
                'openWakeWord',
                'silero-vad v5 (ONNX)',
                'whisper large-v3-turbo',
                'whisper.cpp base.en',
                'Orpheus 3B',
                'Piper voice (en_US-amy)',
                'YOLOv8-worldv2, YOLO11',
                'ArcFace 512 (buffalo_l)',
                'ECAPA 512 (VoxCeleb)',
                'CLIP object embedder',
                'Depth Anything V2',
                'nomic-embed-text',
                'gemma4 8B (Ollama)',
            ],
        },
        {
            title: 'LICENCES',
            items: [
                'Apache 2.0 (most Python parts)',
                'MIT (openWakeWord, ollama)',
                'BSD (numpy, opencv, PyTorch)',
                'AGPL (YOLO family, redistribution)',
                'CC-BY-4.0 (some Piper voices)',
                'Gemma terms (LLM weights)',
                'Orpheus terms (upstream card)',
                'Weights are the supply-chain risk',
                'to watch, not the code',
            ],
        },
    ];

    const colW = 440;
    const marginX = (W - columns.length * colW) / 2;

    columns.forEach((c, i) => {
        const x = marginX + i * colW;
        s += label(x, 140, c.title, { size: 12, weight: 'bold', spacing: 3 });
        s += `<line x1="${x}" y1="152" x2="${x + colW - 40}" y2="152" stroke="${INK}" stroke-width="1.2"/>`;
        c.items.forEach((it, k) => {
            const y = 190 + k * 34;
            s += `<circle cx="${x + 8}" cy="${y - 6}" r="3" fill="${INK}"/>`;
            s += label(x + 22, y, it, { size: 10 });
        });
    });

    s += titleBlock('20', 'DEPENDENCIES AND LICENCES', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 21  ·  OPEN ITEMS MAP (an index card wall)
// ═══════════════════════════════════════════════════════════════════════════
function figOpenItems() {
    // Widened, two columns × three rows, cards sized to hold the wrapped
    // notes without spilling into the next card.
    const W = 1500, H = 1100;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    const cards = [
        { t: 'SCHEMA DRIFT',
          n: 'zero/vision/schemas.py and server/vision/shared/schemas.py have diverged. The sync_schemas.py script referenced in the docstring does not exist in the repository.' },
        { t: 'HALF_LIFE_DAYS  NAME',
          n: 'Used as a time constant, not a half life. Rename the key or divide by ln(2) on read; either resolves the surprise for anyone tuning it.' },
        { t: 'FILLER PRE-SYNTH',
          n: 'Aborts silently after two consecutive TTS failures. Add a counter to the status endpoint so the state is visible before it becomes silence.' },
        { t: 'CONTROL PLANE AUTH',
          n: 'The control plane binds 0.0.0.0 with no auth, a deliberate trade for a LAN only device. If the deployment target changes, add a signed token in the header set.' },
        { t: 'VAD DEFAULT vs SHIPPED',
          n: 'Factory defaults to silero; config.yaml overrides to webrtc. A reader of the factory in isolation draws the wrong conclusion. Reflect the choice in the default.' },
        { t: 'EPISODE CONSOLIDATION',
          n: 'The nightly consolidation job is scaffolded but off by default. Its win over ordinary memory writes has not been measured yet on this subsystem.' },
    ];

    function wrap(text, maxChars) {
        const words = text.split(' ');
        const out = [];
        let cur = '';
        for (const w of words) {
            if ((cur + ' ' + w).trim().length > maxChars) { out.push(cur.trim()); cur = w; }
            else cur += ' ' + w;
        }
        if (cur.trim()) out.push(cur.trim());
        return out;
    }

    const perRow = 2;
    const cellW = 660, cellH = 260;
    const gapX = 40, gapY = 30;
    const totalW = perRow * cellW + (perRow - 1) * gapX;
    const marginX = (W - totalW) / 2;

    cards.forEach((c, i) => {
        const col = i % perRow;
        const row = Math.floor(i / perRow);
        const x = marginX + col * (cellW + gapX);
        const y = 150 + row * (cellH + gapY);

        // Card shell with an accent left rule
        s += `<rect x="${x + 8}" y="${y + 8}" width="${cellW}" height="${cellH}" fill="none" stroke="${MID}" stroke-width="0.5"/>`;
        s += `<rect x="${x}" y="${y}" width="${cellW}" height="${cellH}" fill="none" stroke="${INK}" stroke-width="1.6"/>`;
        s += `<rect x="${x}" y="${y}" width="14" height="${cellH}" fill="${ACC}" opacity="0.7"/>`;

        // Title
        s += label(x + 34, y + 40, c.t, { size: 13, weight: 'bold', spacing: 2 });
        s += `<line x1="${x + 34}" y1="${y + 54}" x2="${x + cellW - 24}" y2="${y + 54}" stroke="${INK}" stroke-width="0.8"/>`;

        // Body, wrapped so it stays inside the card
        const lines = wrap(c.n, 62);
        lines.forEach((ln, k) => {
            s += label(x + 34, y + 90 + k * 22, ln, { size: 10 });
        });
    });

    s += titleBlock('21', 'OPEN ITEMS AND DECISIONS', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// FIGURE 22  ·  DATA FLOW WATERFALL
// A step by step walk of information through the system, from mic frame in
// to speaker sample out, with the data type on every hop and the process
// that carries it. This is the sheet a new reader should be handed first.
// ═══════════════════════════════════════════════════════════════════════════
function figDataFlow() {
    const W = 1500, H = 1600;
    let s = svgHead(W, H);
    s += borderFrame(W, H);

    // Column layout: LEFT rail (input), MIDDLE rail (compute), RIGHT rail
    // (output), plus a persistence panel at the bottom that steps 4 and 12
    // both hit.
    const leftX = 200, midX = 750, rightX = 1300;
    const startY = 130, stepH = 110;

    // Draw a step block
    const step = (rail, i, num, name, dataIn, dataOut, notes) => {
        const y = startY + i * stepH;
        // Rail x
        const x = rail === 'L' ? leftX : rail === 'M' ? midX : rightX;
        // Node
        s += `<rect x="${x - 200}" y="${y}" width="400" height="80" fill="none" stroke="${INK}" stroke-width="1.6"/>`;
        s += `<line x1="${x - 200}" y1="${y + 26}" x2="${x + 200}" y2="${y + 26}" stroke="${INK}" stroke-width="0.9"/>`;
        // Number in a chip on the left
        s += `<rect x="${x - 200}" y="${y}" width="42" height="26" fill="${ACC}"/>`;
        s += label(x - 179, y + 18, String(num).padStart(2, '0'),
            { size: 11, weight: 'bold', color: 'FFFFFF', anchor: 'middle', spacing: 1 });
        // Step name
        s += label(x - 148, y + 18, name, { size: 11, weight: 'bold' });
        // Data in / out
        s += label(x - 192, y + 44, `IN   ${dataIn}`, { size: 8, color: MID, weight: 'bold' });
        s += label(x - 192, y + 62, `OUT  ${dataOut}`, { size: 8, color: MID, weight: 'bold' });
        // Right side note
        s += label(x + 8, y + 44, notes[0] || '', { size: 8 });
        s += label(x + 8, y + 62, notes[1] || '', { size: 8 });
        return y;
    };

    // Arrow between two steps (with data label on the hop)
    const hop = (fromRail, fromI, toRail, toI, dataLabel) => {
        const fromX = fromRail === 'L' ? leftX : fromRail === 'M' ? midX : rightX;
        const toX   = toRail   === 'L' ? leftX : toRail   === 'M' ? midX : rightX;
        const fromY = startY + fromI * stepH + 80;
        const toY   = startY + toI * stepH;
        // If same column, draw straight down
        if (fromRail === toRail) {
            s += `<line x1="${fromX}" y1="${fromY}" x2="${toX}" y2="${toY}" stroke="${INK}" stroke-width="1.4"/>`;
            // Small arrow head
            s += `<polygon points="${toX - 5},${toY - 8} ${toX + 5},${toY - 8} ${toX},${toY}" fill="${INK}"/>`;
            if (dataLabel) s += label(fromX + 12, (fromY + toY) / 2, dataLabel, { size: 8, color: MID, italic: true });
        } else {
            // Elbowed path from one column to another
            const midY = (fromY + toY) / 2;
            s += `<path d="M ${fromX} ${fromY} L ${fromX} ${midY} L ${toX} ${midY} L ${toX} ${toY}"
                    fill="none" stroke="${INK}" stroke-width="1.4"/>`;
            s += `<polygon points="${toX - 5},${toY - 8} ${toX + 5},${toY - 8} ${toX},${toY}" fill="${INK}"/>`;
            if (dataLabel) s += label((fromX + toX) / 2, midY - 6, dataLabel, { size: 8, color: MID, italic: true, anchor: 'middle' });
        }
    };

    // Column headers
    s += label(leftX,  95, 'INPUT SIDE  ·  PI', { size: 11, anchor: 'middle', spacing: 3, weight: 'bold' });
    s += label(midX,   95, 'COMPUTE  ·  PI + GPU NODE', { size: 11, anchor: 'middle', spacing: 3, weight: 'bold' });
    s += label(rightX, 95, 'OUTPUT SIDE  ·  PI', { size: 11, anchor: 'middle', spacing: 3, weight: 'bold' });

    // The 12 steps, laid out in three columns
    step('L', 0, 1,  'MIC CAPTURE',
         'analog audio',  '30 ms int16 frames',
         ['PortAudio callback thread, 16 kHz mono', 'a bounded queue every consumer reads from']);
    step('L', 1, 2,  'WAKE DETECT (idle)',
         '30 ms frames',  'wake accept event',
         ['openWakeWord scores each frame', 'above wake.threshold moves to LISTENING']);
    step('L', 2, 3,  'ENDPOINT (VAD)',
         '30 ms frames',  'utterance PCM',
         ['webrtc VAD as shipped', 'silence_ms trailing silence commits']);
    step('L', 3, 4,  'SPECULATIVE STT KICK',
         'audio so far',  'partial transcript',
         ['fires 180 ms into the silence', 'overlaps STT with the remaining wait']);

    step('M', 4, 5,  'FINAL STT',
         'utterance WAV',  'final transcript',
         ['POST /transcribe on :9000 (Whisper)', 'or whisper.cpp locally if degraded']);
    step('M', 5, 6,  'IDENTITY + DIARIZE',
         'utterance WAV',  'person_id + score',
         ['ECAPA voice embedding, ArcFace off frame', 'fused score against threshold 0.50']);
    step('M', 6, 7,  'RECALL',
         'transcript',    'k memory rows',
         ['under a hard 300 ms budget', 'ranked by relevance x recency x importance']);
    step('M', 7, 8,  'LLM STREAM',
         'prompt tokens', 'reply tokens',
         ['POST /api/chat on :11435 (gemma4)', 'started under the filler for overlap']);

    step('R', 8, 9,  'TTS PRODUCER',
         'sentence text', '24 kHz audio pieces',
         ['POST /tts on :9100 (Orpheus)', 'or Piper locally if degraded']);
    step('R', 9, 10, 'PLAYBACK',
         'audio pieces',  'analog speaker signal',
         ['single gapless output stream', 'sentence by sentence, no inter gap']);
    step('R', 10, 11,'BARGE-IN WATCH',
         '30 ms frames',  'interrupt event',
         ['runs the whole time the robot is speaking', 'wake word or sustained echo aware speech']);
    step('R', 11, 12,'PERSIST',
         'session record','SQLite rows + JSONL',
         ['per-speaker split on end of session', 'reward tagged episode + corpus append']);

    // Hops that carry data between the steps
    hop('L', 0, 'L', 1);
    hop('L', 1, 'L', 2, 'wake accepted');
    hop('L', 2, 'L', 3, 'utterance PCM');
    hop('L', 3, 'M', 4, 'audio → /transcribe');
    hop('M', 4, 'M', 5, 'utterance WAV');
    hop('M', 5, 'M', 6, 'person_id');
    hop('M', 6, 'M', 7, 'transcript + memory');
    hop('M', 7, 'R', 8, 'sentence stream');
    hop('R', 8, 'R', 9, 'audio pieces');
    hop('R', 9, 'R', 10, 'while playing');
    hop('R', 10, 'R', 11, 'session end');

    // Persistence lane (step 12) feeds a footer panel
    const pY = startY + 12 * stepH + 40;
    s += `<rect x="200" y="${pY}" width="${W - 400}" height="90" fill="none" stroke="${INK}" stroke-width="1.4"/>`;
    s += `<line x1="200" y1="${pY + 26}" x2="${W - 200}" y2="${pY + 26}" stroke="${INK}" stroke-width="0.8"/>`;
    s += label(214, pY + 18, 'PERSISTENCE  ·  written on end of session', { size: 11, weight: 'bold', spacing: 2 });
    s += label(214, pY + 46, '·  zero_memory.sqlite  ·  facts credited to person_id if score ≥ write_min_score', { size: 9 });
    s += label(214, pY + 64, '·  zero_episodes.sqlite  ·  turn episode with reward and surprise tag', { size: 9 });
    s += label(214, pY + 82, '·  data/corpus/interactions.jsonl  ·  per-speaker session record for offline training', { size: 9 });

    // Legend
    const legY = pY + 130;
    s += label(200, legY, 'THIS DIAGRAM IS THE HAPPY PATH.', { size: 10, spacing: 2, weight: 'bold' });
    s += label(200, legY + 22, 'Failure branches from any GPU step degrade to their local fallback described in Figure 6, and the degraded flag is', { size: 9, color: MID });
    s += label(200, legY + 38, 'appended to the persona prompt so the robot can say so honestly rather than becoming silently worse.', { size: 9, color: MID });

    s += titleBlock('22', 'DATA FLOW  ·  A TURN, END TO END', W - 480, H - 160);
    s += `</svg>`;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// COVER ART  ·  full AF-1 unit shown as an engineering elevation
// ═══════════════════════════════════════════════════════════════════════════
function figCover() {
    const W = 1000, H = 1250;
    let s = svgHead(W, H);
    // Fine frame + drafting corners
    s += borderFrame(W, H);

    // Title lockup
    s += label(80, 80, 'ZEROBIONIC', { size: 22, spacing: 6, weight: 'bold' });
    s += label(80, 106, 'AF-1 ASSISTIVE HUMANOID PROGRAM', { size: 9, spacing: 4, color: MID });
    s += `<line x1="80" y1="120" x2="${W - 80}" y2="120" stroke="${INK}" stroke-width="1.4"/>`;

    // Big centred AF-1 elevation (upper body only, matching the head cutaway)
    const cx = 500, cy = 640;

    // Body plate + shoulders
    s += `
      <!-- shoulders -->
      <path d="M ${cx - 260} ${cy + 140}
               C ${cx - 260} ${cy + 100}, ${cx - 230} ${cy + 70}, ${cx - 180} ${cy + 70}
               L ${cx + 180} ${cy + 70}
               C ${cx + 230} ${cy + 70}, ${cx + 260} ${cy + 100}, ${cx + 260} ${cy + 140}
               L ${cx + 260} ${cy + 400}
               L ${cx - 260} ${cy + 400} Z"
            fill="none" stroke="${INK}" stroke-width="2"/>
      <!-- collar detail -->
      <line x1="${cx - 180}" y1="${cy + 70}" x2="${cx - 140}" y2="${cy + 110}" stroke="${INK}" stroke-width="1"/>
      <line x1="${cx + 180}" y1="${cy + 70}" x2="${cx + 140}" y2="${cy + 110}" stroke="${INK}" stroke-width="1"/>
      <line x1="${cx - 140}" y1="${cy + 110}" x2="${cx + 140}" y2="${cy + 110}" stroke="${INK}" stroke-width="1"/>
      <!-- chest emblem -->
      <circle cx="${cx}" cy="${cy + 200}" r="34" fill="none" stroke="${INK}" stroke-width="1.4"/>
      <circle cx="${cx}" cy="${cy + 200}" r="22" fill="none" stroke="${INK}" stroke-width="1"/>
      <circle cx="${cx}" cy="${cy + 200}" r="8" fill="${INK}"/>
      ${label(cx, cy + 260, 'AF-1', { size: 9, anchor: 'middle', spacing: 4, weight: 'bold' })}
      <!-- side plating hints -->
      <line x1="${cx - 220}" y1="${cy + 170}" x2="${cx - 220}" y2="${cy + 380}" stroke="${INK}" stroke-width="0.8"/>
      <line x1="${cx + 220}" y1="${cy + 170}" x2="${cx + 220}" y2="${cy + 380}" stroke="${INK}" stroke-width="0.8"/>
    `;

    // Head
    s += `
      <path d="
        M ${cx - 140} ${cy - 200}
        C ${cx - 180} ${cy - 180}, ${cx - 200} ${cy - 120}, ${cx - 195} ${cy - 60}
        C ${cx - 190} ${cy + 0}, ${cx - 180} ${cy + 30}, ${cx - 150} ${cy + 55}
        L ${cx + 150} ${cy + 55}
        C ${cx + 180} ${cy + 30}, ${cx + 190} ${cy + 0}, ${cx + 195} ${cy - 60}
        C ${cx + 200} ${cy - 120}, ${cx + 180} ${cy - 180}, ${cx + 140} ${cy - 200}
        Z"
        fill="none" stroke="${INK}" stroke-width="2"/>
      <line x1="${cx - 180}" y1="${cy - 90}" x2="${cx + 180}" y2="${cy - 90}" stroke="${INK}" stroke-width="1.4"/>
      <line x1="${cx - 180}" y1="${cy - 70}" x2="${cx + 180}" y2="${cy - 70}" stroke="${INK}" stroke-width="1.4"/>
      <circle cx="${cx - 100}" cy="${cy - 80}" r="10" fill="none" stroke="${INK}" stroke-width="1.4"/>
      <circle cx="${cx - 100}" cy="${cy - 80}" r="5" fill="${INK}"/>
      <circle cx="${cx + 100}" cy="${cy - 80}" r="10" fill="none" stroke="${INK}" stroke-width="1.4"/>
      <circle cx="${cx + 100}" cy="${cy - 80}" r="5" fill="${INK}"/>
      <!-- mic grille -->
      ${Array.from({length: 12}, (_, i) => `<line x1="${cx - 60 + i * 10}" y1="${cy + 15}" x2="${cx - 60 + i * 10}" y2="${cy + 40}" stroke="${INK}" stroke-width="1"/>`).join('')}
      <!-- crown antenna -->
      <line x1="${cx - 10}" y1="${cy - 210}" x2="${cx - 5}" y2="${cy - 270}" stroke="${INK}" stroke-width="1.4"/>
      <circle cx="${cx - 5}" cy="${cy - 270}" r="4" fill="none" stroke="${INK}" stroke-width="1.4"/>
      <line x1="${cx - 5}" y1="${cy - 270}" x2="${cx - 5}" y2="${cy - 300}" stroke="${INK}" stroke-width="1.4"/>
    `;

    // Callouts (subtle, faint)
    s += callout(cx - 100, cy - 80, 120, cy - 240, 'STEREO APERTURES', { color: MID, size: 8 });
    s += callout(cx + 100, cy - 80, W - 250, cy - 240, 'FACE EMBEDDER FEEDS', { color: MID, size: 8 });
    s += callout(cx, cy + 27, W - 250, cy - 100, 'MIC GRILLE  ·  16 kHz MONO', { color: MID, size: 8 });
    s += callout(cx, cy + 200, 120, cy + 260, 'CHEST EMBLEM', { color: MID, size: 8 });

    // Bottom lockup
    s += label(80, H - 210, 'ZERO  ·  BRAIN BUILD  ·  INTELLIGENCE STACK', { size: 20, spacing: 4, weight: 'bold' });
    s += label(80, H - 182, 'Conversational Subsystem for the AF-1 Assistive Robotics Platform',
        { size: 10, color: MID, family: 'Constantia, serif', italic: true });

    s += `<line x1="80" y1="${H - 160}" x2="${W - 80}" y2="${H - 160}" stroke="${INK}" stroke-width="1"/>`;

    // Metadata block
    s += label(80, H - 140, 'DOCUMENT', { size: 7, spacing: 2, color: MID });
    s += label(80, H - 126, 'ZBR-AF1-SW-001', { size: 10 });
    s += label(80, H - 108, 'REVISION', { size: 7, spacing: 2, color: MID });
    s += label(80, H - 94, '0.1  ·  first public draft', { size: 10 });

    s += label(340, H - 140, 'PREPARED BY', { size: 7, spacing: 2, color: MID });
    s += label(340, H - 126, 'Denis Obila', { size: 10 });
    s += label(340, H - 108, 'ROLE', { size: 7, spacing: 2, color: MID });
    s += label(340, H - 94, 'ZERO Systems Lead, Zerobionic', { size: 10 });

    s += label(640, H - 140, 'REPOSITORY', { size: 7, spacing: 2, color: MID });
    s += label(640, H - 126, 'git commit as tagged in Section 12', { size: 10 });
    s += label(640, H - 108, 'TARGET HARDWARE', { size: 7, spacing: 2, color: MID });
    s += label(640, H - 94, 'AF-1 head Pi 5 + GPU node', { size: 10 });

    s += label(80, H - 60, 'CONFIDENTIAL  ·  INTERNAL ENGINEERING DOCUMENT', { size: 8, spacing: 4, color: MID });

    s += `</svg>`;
    return s;
}

// ── Render loop ──────────────────────────────────────────────────────────
const figures = [
    { file: 'v2-cover.svg',       fn: figCover,           png: 'v2-cover.png',       widthPx: 2400 },
    { file: 'v2-fig-01.svg',      fn: figHeadCutaway,     png: 'v2-fig-01.png',      widthPx: 2400 },
    { file: 'v2-fig-02.svg',      fn: figNervousSystem,   png: 'v2-fig-02.png',      widthPx: 2400 },
    { file: 'v2-fig-03.svg',      fn: figProcessHive,     png: 'v2-fig-03.png',      widthPx: 2400 },
    { file: 'v2-fig-04.svg',      fn: figTurnTiming,      png: 'v2-fig-04.png',      widthPx: 2400 },
    { file: 'v2-fig-05.svg',      fn: figTiers,           png: 'v2-fig-05.png',      widthPx: 2400 },
    { file: 'v2-fig-06.svg',      fn: figFallback,        png: 'v2-fig-06.png',      widthPx: 2400 },
    { file: 'v2-fig-07.svg',      fn: figControlSurface,  png: 'v2-fig-07.png',      widthPx: 2400 },
    { file: 'v2-fig-08.svg',      fn: figStores,          png: 'v2-fig-08.png',      widthPx: 2400 },
    { file: 'v2-fig-09.svg',      fn: figEndpoint,        png: 'v2-fig-09.png',      widthPx: 2400 },
    { file: 'v2-fig-10.svg',      fn: figIdentity,        png: 'v2-fig-10.png',      widthPx: 2400 },
    { file: 'v2-fig-11.svg',      fn: figDecay,           png: 'v2-fig-11.png',      widthPx: 2400 },
    { file: 'v2-fig-12.svg',      fn: figBuildAssembly,   png: 'v2-fig-12.png',      widthPx: 2400 },
    { file: 'v2-fig-13.svg',      fn: figTestRig,         png: 'v2-fig-13.png',      widthPx: 2400 },
    { file: 'v2-fig-14.svg',      fn: figFailureBreakers, png: 'v2-fig-14.png',      widthPx: 2400 },
    { file: 'v2-fig-15.svg',      fn: figConfigTree,      png: 'v2-fig-15.png',      widthPx: 2400 },
    { file: 'v2-fig-16.svg',      fn: figLatency,         png: 'v2-fig-16.png',      widthPx: 2400 },
    { file: 'v2-fig-17.svg',      fn: figPrivacy,         png: 'v2-fig-17.png',      widthPx: 2400 },
    { file: 'v2-fig-18.svg',      fn: figObservability,   png: 'v2-fig-18.png',      widthPx: 2400 },
    { file: 'v2-fig-19.svg',      fn: figRunbook,         png: 'v2-fig-19.png',      widthPx: 2400 },
    { file: 'v2-fig-20.svg',      fn: figDependencies,    png: 'v2-fig-20.png',      widthPx: 2400 },
    { file: 'v2-fig-21.svg',      fn: figOpenItems,       png: 'v2-fig-21.png',      widthPx: 2400 },
    { file: 'v2-fig-22.svg',      fn: figDataFlow,        png: 'v2-fig-22.png',      widthPx: 2400 },
];

for (const f of figures) {
    const svg = f.fn();
    fs.writeFileSync(path.join(OUT, f.file), svg);
    try {
        const r = new Resvg(svg, { fitTo: { mode: 'width', value: f.widthPx }, background: 'rgba(0,0,0,0)' });
        fs.writeFileSync(path.join(OUT, f.png), r.render().asPng());
        console.log('rendered', f.png);
    } catch (e) {
        console.error('FAILED', f.file, e.message);
    }
}
