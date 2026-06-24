const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType, VerticalAlign,
} = require("docx");

const CONTENT_W = 9360; // US Letter, 1" margins
const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const cellMargins = { top: 60, bottom: 60, left: 120, right: 120 };

function headerCell(text, width) {
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA }, margins: cellMargins,
    shading: { fill: "1F4E79", type: ShadingType.CLEAR },
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ children: [new TextRun({ text, bold: true, color: "FFFFFF", size: 18 })] })],
  });
}
function bodyCell(text, width, opts = {}) {
  const runs = Array.isArray(text)
    ? text
    : [new TextRun({ text, size: 18, bold: opts.bold || false })];
  return new TableCell({
    borders, width: { size: width, type: WidthType.DXA }, margins: cellMargins,
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    children: [new Paragraph({ children: runs })],
  });
}
function row(cells) { return new TableRow({ children: cells }); }

function table(widths, headerLabels, dataRows) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      row(headerLabels.map((l, i) => headerCell(l, widths[i]))),
      ...dataRows.map((r) => row(r.map((c, i) =>
        typeof c === "object" && c.runs
          ? bodyCell(c.runs, widths[i], c)
          : bodyCell(c, widths[i], { fill: r.__fill })))),
    ],
  });
}

const stack = table(
  [2600, 4360, 2400],
  ["Stage", "Engine", "Status"],
  [
    ["Wake word", "openWakeWord (“Hey Jarvis”, ONNX runtime)", "Working"],
    ["Voice activity / endpointing", "silero-VAD", "Working"],
    ["Speech-to-text", "whisper.cpp (tiny.en)", "~2 s / utterance"],
    ["Conversation brain", "Ollama + Llama 3.2 3B", "Working"],
    ["Text-to-speech", "Piper (Fish S1-mini planned for emotion)", "Speaking"],
    ["Audio I/O", "Logitech BRIO USB mic in · Bluetooth speaker out", "Working"],
  ]
);

const gpu = table(
  [3000, 3180, 3180],
  ["Factor", "Raspberry Pi 5 (CPU only)", "GPU host"],
  [
    ["LLM reply (3B model)", "~9–12 s / turn (a few tokens/sec)", "< 1–2 s (50–100+ tokens/sec)"],
    ["Speech-to-text", "~2–6 s", "~0.2–0.5 s (5–20× faster)"],
    ["Expressive TTS (Fish S1-mini)", "Impractical on CPU", "Real-time — unlocks laughs / sighs"],
    ["Memory", "8 GB shared; can’t hold all models", "VRAM + RAM hold all models at once"],
    ["Sustained load", "Thermal throttling", "Designed for it"],
    ["Model headroom", "Small models only", "7B+ LLMs, vision, humanoid features"],
  ]
);

function h1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] });
}
function p(runs, opts = {}) {
  return new Paragraph({ spacing: { after: 120 }, ...opts,
    children: Array.isArray(runs) ? runs : [new TextRun({ text: runs, size: 20 })] });
}
function bullet(text) {
  return new Paragraph({ numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60 }, children: [new TextRun({ text, size: 20 })] });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, color: "1F4E79", font: "Arial" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 0 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: "bullet", text: "•",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 460, hanging: 260 } } } }] },
    ],
  },
  sections: [{
    properties: { page: {
      size: { width: 12240, height: 15840 },
      margin: { top: 1080, right: 1440, bottom: 1080, left: 1440 },
    } },
    children: [
      new Paragraph({ spacing: { after: 40 },
        children: [new TextRun({ text: "ZERO — Progress Report", bold: true, size: 36, color: "1F4E79" })] }),
      new Paragraph({ spacing: { after: 160 }, children: [new TextRun({
        text: "24 June 2026   ·   Platform: Raspberry Pi 5 (8 GB), fully offline   ·   Author: Gregory Kimani",
        size: 18, color: "555555" })] }),

      h1("1. Objective"),
      p("Build the first step toward a conversational humanoid: a device you talk to in English that replies in a natural human voice — running entirely on-device, with no internet."),

      h1("2. What We Built Today"),
      p("A complete, working offline voice assistant with an interface-first, config-driven architecture — every stage is swappable via config.yaml with no code changes."),
      stack,
      p([new TextRun({ text: "Key features delivered.  ", bold: true, size: 20 }),
         new TextRun({ text: "Multi-turn conversation mode (wake once, converse freely, sleep after 30 s of silence); streaming replies (speaks each sentence while the LLM generates the rest); and diagnostic tooling for audio devices, mic level, wake score, and TTS.", size: 20 })],
        { spacing: { before: 120, after: 80 } }),
      p([new TextRun({ text: "Problems solved.  ", bold: true, size: 20 }),
         new TextRun({ text: "openWakeWord install (switched to ONNX for Python 3.12/ARM); whisper model loading; silero VAD 512-sample chunking; audio-queue overflow; and whisper hallucinations on silence (VAD threshold + minimum-speech gate).", size: 20 })]),

      h1("3. Result"),
      p("A genuinely usable offline assistant: wake → natural back-and-forth → sleep, with ~2 s transcription and spoken replies through the Bluetooth speaker. The full pipeline is proven end-to-end on the Pi 5."),

      h1("4. Why a GPU Is the Better Long-Term Host"),
      p("The Pi 5 has no ML-usable GPU — every model runs on its 4 CPU cores. This is the root cause of our main limitation: latency."),
      gpu,
      p([new TextRun({ text: "Recommendation.  ", bold: true, size: 20 }),
         new TextRun({ text: "The Pi 5 is excellent for proving the system and for low-power, portable deployment — and it has succeeded at that. But the project’s defining goals (near-instant conversation and a genuinely human, expressive voice) are gated by compute the Pi cannot provide. A GPU host — or a hybrid where the Pi is the edge device talking to a GPU server — removes the latency wall, makes expressive TTS viable, and leaves room for the vision and larger-model work the humanoid roadmap will need.", size: 20 })],
        { spacing: { before: 120 } }),

      h1("5. Next Steps"),
      bullet("Enable the expressive voice (Fish S1-mini) once a GPU is available; benchmark real-time factor."),
      bullet("Train a custom “Hey Zero” wake word to replace the placeholder “Hey Jarvis”."),
      bullet("Optional: add a short mic-mute window after speaking to prevent speaker-into-mic echo."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("docs/ZERO_Progress_Report_2026-06-24.docx", buf);
  console.log("wrote docs/ZERO_Progress_Report_2026-06-24.docx");
});
