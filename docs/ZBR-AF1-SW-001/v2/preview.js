const mammoth = require('mammoth');
const fs = require('fs');
mammoth.convertToHtml({ path: 'C:\\Users\\Allela_Work\\Desktop\\ZERO\\docs\\ZBR-AF1-SW-001\\ZBR-AF1-SW-001_v2.docx' })
    .then(r => {
        const css = `body{max-width:820px;margin:40px auto;font-family:Constantia,serif;color:#25231B;line-height:1.55;padding:20px}
img{max-width:100%;height:auto;display:block;margin:20px auto}
h1,h2,h3{font-family:"Franklin Gothic Heavy",sans-serif}
p{text-align:justify}`;
        fs.writeFileSync(
            'C:\\Users\\Allela_Work\\Desktop\\ZERO\\docs\\ZBR-AF1-SW-001\\v2\\preview_v2.html',
            `<html><head><meta charset="utf-8"><style>${css}</style></head><body>${r.value}</body></html>`
        );
        console.log('preview_v2.html written, warnings:', r.messages.length);
    })
    .catch(e => { console.error(e); process.exit(1); });
