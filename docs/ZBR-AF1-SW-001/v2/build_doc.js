// Builds the final ZBR-AF1-SW-001 document with proper title page, TOC,
// consistent typography, Roman numeral subsections, illustrations placed
// inline with prose. Written to `../ZBR-AF1-SW-001_v2.docx`.

const fs = require('fs');
const path = require('path');
const docx = require('docx');
const prose = require('./prose');

const {
    Document, Packer, Paragraph, TextRun, ImageRun,
    HeadingLevel, AlignmentType, TabStopPosition, TabStopType,
    PageBreak, convertInchesToTwip, BorderStyle, PositionalTab,
    PositionalTabAlignment, PositionalTabLeader,
    Header, Footer, PageNumber, NumberFormat, LineRuleType,
    SectionType,
} = docx;

const BASE = __dirname;
const OUT  = path.join(BASE, '..', 'ZBR-AF1-SW-001_v3.docx');

// ── Palette ──────────────────────────────────────────────────────────────
const INK  = '25231B';
const MID  = '8A8172';
const RULE = 'D9D4C7';
const ACC  = 'B85425';    // burnt orange, matches the cover accent

// ── Fonts ────────────────────────────────────────────────────────────────
// Body prose:   Constantia (serif, warm, reads at 10.5 pt)
// Headings:     Franklin Gothic Heavy (display sans, bold)
// Small caps / tags:  Courier New
const FONT_BODY = 'Constantia';
const FONT_HEAD = 'Franklin Gothic Heavy';
const FONT_MONO = 'Courier New';

// ── Prose paragraph ──────────────────────────────────────────────────────
function bodyPara(text) {
    return new Paragraph({
        spacing: { after: 180, line: 300, lineRule: LineRuleType.AUTO },
        alignment: AlignmentType.JUSTIFIED,
        children: [
            new TextRun({
                text: text.replace(/\s+/g, ' ').trim(),
                font: FONT_BODY,
                size: 22,          // 11 pt
                color: INK,
            }),
        ],
    });
}

// ── Section heading (top-level) ──────────────────────────────────────────
function sectionHeading(num, title) {
    return [
        new Paragraph({
            spacing: { before: 200, after: 40 },
            pageBreakBefore: true,
            children: [
                new TextRun({
                    text: `SECTION ${num}`,
                    font: FONT_MONO,
                    size: 18,     // 9 pt small tag
                    color: ACC,
                    characterSpacing: 80,
                }),
            ],
        }),
        new Paragraph({
            spacing: { after: 300 },
            border: {
                bottom: { style: BorderStyle.SINGLE, size: 4, color: INK, space: 4 },
            },
            children: [
                new TextRun({
                    text: title,
                    font: FONT_HEAD,
                    size: 44,     // 22 pt
                    color: INK,
                    bold: true,
                }),
            ],
        }),
    ];
}

// ── Subsection heading (Roman numeral) ───────────────────────────────────
function subHeading(roman, title) {
    return new Paragraph({
        spacing: { before: 400, after: 160 },
        keepNext: true,
        children: [
            new TextRun({
                text: `${roman}.  `,
                font: FONT_HEAD,
                size: 26,   // 13 pt
                color: ACC,
                bold: true,
            }),
            new TextRun({
                text: title,
                font: FONT_HEAD,
                size: 26,   // 13 pt
                color: INK,
                bold: true,
            }),
        ],
    });
}

// ── Figure inline with caption ───────────────────────────────────────────
function figure(num) {
    const pngPath = path.join(BASE, `v2-fig-${String(num).padStart(2, '0')}.png`);
    if (!fs.existsSync(pngPath)) {
        return [bodyPara(`[figure ${num} missing]`)];
    }
    const data = fs.readFileSync(pngPath);
    const w = data.readUInt32BE(16);
    const h = data.readUInt32BE(20);
    const aspect = h / w;
    const widthIn = 6.6;
    return [
        new Paragraph({
            spacing: { before: 240, after: 60 },
            alignment: AlignmentType.CENTER,
            children: [
                new ImageRun({
                    data,
                    transformation: {
                        width: Math.round(widthIn * 96),
                        height: Math.round(widthIn * 96 * aspect),
                    },
                    type: 'png',
                }),
            ],
        }),
        new Paragraph({
            spacing: { after: 40 },
            alignment: AlignmentType.CENTER,
            children: [
                new TextRun({
                    text: `Figure ${num}`,
                    font: FONT_MONO,
                    size: 16,
                    color: MID,
                    characterSpacing: 120,
                }),
            ],
        }),
        new Paragraph({
            spacing: { after: 320 },
            alignment: AlignmentType.CENTER,
            border: {
                bottom: { style: BorderStyle.SINGLE, size: 4, color: RULE, space: 8 },
            },
            children: [
                new TextRun({
                    text: figureCaption(num),
                    font: FONT_BODY,
                    size: 20,
                    color: MID,
                    italics: true,
                }),
            ],
        }),
    ];
}

function figureCaption(num) {
    return {
        1: 'AF-1 head, anatomical cutaway. Physical components map onto the software function they carry.',
        2: 'Two-machine nervous system. One tunnel binds the AF-1 head to the GPU node.',
        3: 'The process set as a hive around the main loop. Persistent threads run for the life of the process; transient threads live and die inside a turn.',
        4: 'A single turn on one time axis. Every overlap is deliberate and was worth writing threading code for.',
        5: 'Three perception tiers over a twelve second window that goes from moving to still.',
        6: 'The fallback pattern drawn once. Six faculties share it. The local contact stays dashed until the first remote failure builds it.',
        7: 'The control plane surface. Inbound endpoints on the left, outbound tunnel calls on the right, one brain in the middle.',
        8: 'Persistent stores at a glance, with the sign convention that determines who owns each row.',
        9: 'The endpoint decision ladder. The tri-state hold is the only branch that can extend a wait.',
        10: 'Identity decision surface. Weak but agreeing signals accept where weak conflicting ones do not.',
        11: 'Memory recency contribution against age. The 0.2 floor keeps an old but important fact reachable.',
        12: 'Build assembly, exploded. The Pi head stack and the GPU node stack are two separate deployments joined by one tunnel.',
        13: 'The test rig. A known utterance, a scope for turn timing, a stopwatch for end to end.',
        14: 'Failure triage as a breaker panel. Each failure has an explicit trip and an explicit response.',
        15: 'Configuration anatomy. Every top level block is a faculty; every leaf is documented in place.',
        16: 'Latency waterfall for a healthy turn, alongside the Pi 5 power envelope.',
        17: 'Privacy layers. Enrolled at the centre, guarded by default, strict for hostile settings.',
        18: 'A session read from the logs. The named lines are load bearing for triage.',
        19: 'Operations runbook. Bring up on the left, triage in the middle, deploy and backup on the right.',
        20: 'Dependencies and licences. The code tree is boring and safe; the weights tree is where attention has to be paid.',
        21: 'Open items and design decisions worth recording.',
        22: 'Data flow through the system, one turn end to end. Every hop carries a specific data type between named processes.',
    }[num] || '';
}

// ── TOC line ─────────────────────────────────────────────────────────────
function tocLine(numLabel, title, sub = false) {
    return new Paragraph({
        spacing: { after: sub ? 40 : 100 },
        indent: sub ? { left: 480 } : undefined,
        children: [
            new TextRun({
                text: `${numLabel}   `,
                font: FONT_MONO,
                size: sub ? 18 : 20,
                color: sub ? MID : INK,
                bold: !sub,
                characterSpacing: 60,
            }),
            new TextRun({
                text: title,
                font: FONT_BODY,
                size: sub ? 20 : 22,
                color: sub ? MID : INK,
                bold: !sub,
            }),
        ],
    });
}

// ── Render a whole section from its prose array ──────────────────────────
function renderSection(num, contents) {
    const out = [];
    // First entry is H('title')
    const heading = contents.find(c => c && c.kind === 'sec');
    out.push(...sectionHeading(num, heading.title));

    for (const c of contents) {
        if (typeof c === 'string') {
            out.push(bodyPara(c));
        } else if (c && c.kind === 'sub') {
            out.push(subHeading(c.roman, c.title));
        } else if (c && c.kind === 'fig') {
            out.push(...figure(c.num));
        }
    }
    return out;
}

// ── Cover page ────────────────────────────────────────────────────────────
function coverPage() {
    // The user maintains their own cover design; this text cover is a
    // clean placeholder they can replace with their layout. All fields are
    // rendered as text so the metadata is present either way.
    const runs = [];
    runs.push(
        new Paragraph({
            spacing: { before: 400, after: 60 },
            children: [
                new TextRun({ text: 'ZEROBIONIC  ·  AF-1 ASSISTIVE ROBOTICS  ·  AI / ML / INTELLIGENCE',
                    font: FONT_MONO, size: 18, color: ACC, characterSpacing: 100, bold: true }),
            ],
        }),
        new Paragraph({
            spacing: { after: 40 },
            children: [
                new TextRun({ text: 'ZERO', font: FONT_HEAD, size: 88, color: INK, bold: true }),
            ],
        }),
        new Paragraph({
            spacing: { after: 300 },
            children: [
                new TextRun({ text: 'Brain Build / Intelligence Stack',
                    font: FONT_HEAD, size: 44, color: ACC, bold: true }),
            ],
            border: {
                bottom: { style: BorderStyle.SINGLE, size: 12, color: ACC, space: 12 },
            },
        }),
        new Paragraph({
            spacing: { after: 400 },
            children: [
                new TextRun({
                    text: 'ZERO is the AF-1 conversational intelligence subsystem, a voice brain on a Raspberry Pi 5 that offloads its heavy models to a GPU node over one SSH tunnel. It runs a continuous wake, listen, think and speak loop, with vision, identity, memory and tools as real faculties, each holding a local fallback when the tunnel drops. This document is its reference.',
                    font: FONT_BODY, size: 24, color: INK,
                }),
            ],
        }),
        // Metadata grid
        new Paragraph({
            spacing: { after: 20 },
            children: [
                new TextRun({ text: 'DOCUMENT ID   ',
                    font: FONT_MONO, size: 14, color: MID, characterSpacing: 80, bold: true }),
                new TextRun({ text: 'ZBR-AF1-SW-001',
                    font: FONT_MONO, size: 18, color: INK }),
            ],
        }),
        new Paragraph({
            spacing: { after: 20 },
            children: [
                new TextRun({ text: 'REVISION      ',
                    font: FONT_MONO, size: 14, color: MID, characterSpacing: 80, bold: true }),
                new TextRun({ text: '0.1  ·  first public draft',
                    font: FONT_MONO, size: 18, color: INK }),
            ],
        }),
        new Paragraph({
            spacing: { after: 20 },
            children: [
                new TextRun({ text: 'PREPARED BY   ',
                    font: FONT_MONO, size: 14, color: MID, characterSpacing: 80, bold: true }),
                new TextRun({ text: 'Sam Obila Allela  ·  Machine Learning Lead, Zerobionic',
                    font: FONT_MONO, size: 18, color: INK }),
            ],
        }),
        new Paragraph({
            spacing: { after: 20 },
            children: [
                new TextRun({ text: 'DATE          ',
                    font: FONT_MONO, size: 14, color: MID, characterSpacing: 80, bold: true }),
                new TextRun({ text: '27 July 2026',
                    font: FONT_MONO, size: 18, color: INK }),
            ],
        }),
        new Paragraph({
            spacing: { after: 20 },
            children: [
                new TextRun({ text: 'TARGET HW     ',
                    font: FONT_MONO, size: 14, color: MID, characterSpacing: 80, bold: true }),
                new TextRun({ text: 'AF-1 head Raspberry Pi 5  ·  GPU node (any CUDA class card)',
                    font: FONT_MONO, size: 18, color: INK }),
            ],
        }),
        new Paragraph({
            spacing: { before: 500, after: 40 },
            border: {
                top: { style: BorderStyle.SINGLE, size: 6, color: INK, space: 8 },
            },
            children: [
                new TextRun({
                    text: 'CONFIDENTIAL  ·  INTERNAL ENGINEERING DOCUMENT',
                    font: FONT_MONO, size: 14, color: MID, characterSpacing: 200,
                }),
            ],
        }),
    );
    return runs;
}

// ── Table of Contents ────────────────────────────────────────────────────
function tocPage() {
    const out = [];
    out.push(new Paragraph({
        pageBreakBefore: true,
        spacing: { before: 200, after: 60 },
        children: [
            new TextRun({ text: 'CONTENTS',
                font: FONT_MONO, size: 18, color: ACC, characterSpacing: 200 }),
        ],
    }));
    out.push(new Paragraph({
        spacing: { after: 400 },
        border: {
            bottom: { style: BorderStyle.SINGLE, size: 4, color: INK, space: 6 },
        },
        children: [
            new TextRun({ text: 'Table of Contents',
                font: FONT_HEAD, size: 40, color: INK, bold: true }),
        ],
    }));

    const walk = (secNum, contents) => {
        const heading = contents.find(c => c && c.kind === 'sec');
        out.push(tocLine(String(secNum), heading.title));
        for (const c of contents) {
            if (c && c.kind === 'sub') {
                out.push(tocLine(`${secNum}.${c.roman}`, c.title, true));
            }
        }
    };
    walk(1, prose.section1);
    walk(2, prose.section2);
    walk(3, prose.section3);
    walk(4, prose.section4);
    walk(5, prose.section5);
    walk(6, prose.section6);
    walk(7, prose.section7);
    walk(8, prose.section8);
    walk(9, prose.section9);
    walk(10, prose.section10);
    walk(11, prose.section11);
    walk(12, prose.section12);
    walk(13, prose.section13);
    walk(14, prose.section14);
    // Appendices
    const appHeading = prose.appendices.find(c => c && c.kind === 'sec');
    out.push(tocLine('A', appHeading.title));
    for (const c of prose.appendices) {
        if (c && c.kind === 'sub') {
            out.push(tocLine(`A.${c.roman}`, c.title, true));
        }
    }
    return out;
}

// ── Assemble the document ────────────────────────────────────────────────
async function main() {
    const children = [
        ...coverPage(),
        ...tocPage(),
        ...renderSection(1,  prose.section1),
        ...renderSection(2,  prose.section2),
        ...renderSection(3,  prose.section3),
        ...renderSection(4,  prose.section4),
        ...renderSection(5,  prose.section5),
        ...renderSection(6,  prose.section6),
        ...renderSection(7,  prose.section7),
        ...renderSection(8,  prose.section8),
        ...renderSection(9,  prose.section9),
        ...renderSection(10, prose.section10),
        ...renderSection(11, prose.section11),
        ...renderSection(12, prose.section12),
        ...renderSection(13, prose.section13),
        ...renderSection(14, prose.section14),
        ...renderSection('A', prose.appendices),
    ];

    const doc = new Document({
        creator: 'Sam Obila Allela',
        title: 'ZBR-AF1-SW-001  ·  ZERO Brain Build / Intelligence Stack',
        description: 'Reference document for the ZERO conversational subsystem of the AF-1.',
        styles: {
            default: {
                document: {
                    run: { font: FONT_BODY, size: 22, color: INK },
                    paragraph: { spacing: { line: 300, lineRule: LineRuleType.AUTO } },
                },
            },
        },
        sections: [{
            properties: {
                page: {
                    size: {
                        // A4 portrait
                        width: convertInchesToTwip(8.27),
                        height: convertInchesToTwip(11.69),
                    },
                    margin: {
                        top: convertInchesToTwip(0.9),
                        bottom: convertInchesToTwip(0.9),
                        left: convertInchesToTwip(0.85),
                        right: convertInchesToTwip(0.85),
                        header: convertInchesToTwip(0.4),
                        footer: convertInchesToTwip(0.4),
                    },
                },
            },
            headers: {
                default: new Header({
                    children: [new Paragraph({
                        alignment: AlignmentType.LEFT,
                        children: [
                            new TextRun({ text: 'ZBR-AF1-SW-001',
                                font: FONT_MONO, size: 14, color: MID, characterSpacing: 100 }),
                            new TextRun({ text: '    ·    ZERO  ·  BRAIN BUILD / INTELLIGENCE STACK',
                                font: FONT_MONO, size: 14, color: MID, characterSpacing: 100 }),
                        ],
                    })],
                }),
            },
            footers: {
                default: new Footer({
                    children: [new Paragraph({
                        alignment: AlignmentType.RIGHT,
                        children: [
                            new TextRun({ text: 'PAGE  ',
                                font: FONT_MONO, size: 14, color: MID, characterSpacing: 100 }),
                            new TextRun({ children: [PageNumber.CURRENT],
                                font: FONT_MONO, size: 14, color: MID }),
                            new TextRun({ text: '  /  ',
                                font: FONT_MONO, size: 14, color: MID }),
                            new TextRun({ children: [PageNumber.TOTAL_PAGES],
                                font: FONT_MONO, size: 14, color: MID }),
                        ],
                    })],
                }),
            },
            children,
        }],
    });

    const buf = await Packer.toBuffer(doc);
    fs.writeFileSync(OUT, buf);
    console.log('Wrote', OUT, `(${buf.length} bytes)`);
}

main().catch(err => { console.error(err); process.exit(1); });
