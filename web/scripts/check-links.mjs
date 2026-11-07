// Verifica que cada href/src local de los HTML del sitio resuelva a un archivo
// existente y que cada #ancla exista en la página destino.
// Uso (desde web/):  npm run check:links
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const skipDirs = new Set(['node_modules', 'src', 'scripts']);

function walk(dir) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    if (skipDirs.has(e.name)) continue;
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p));
    else if (e.name.endsWith('.html')) out.push(p);
  }
  return out;
}

const idsCache = new Map();
function idsOf(file) {
  if (!idsCache.has(file)) {
    const html = fs.readFileSync(file, 'utf8');
    idsCache.set(file, new Set([...html.matchAll(/\sid="([^"]+)"/g)].map((m) => m[1])));
  }
  return idsCache.get(file);
}

let errors = 0;
let checked = 0;
for (const file of walk(root)) {
  const html = fs.readFileSync(file, 'utf8').replace(/<!--[\s\S]*?-->/g, '');
  const rel = path.relative(root, file).split(path.sep).join('/');
  for (const m of html.matchAll(/\s(?:href|src)="([^"]+)"/g)) {
    const url = m[1];
    if (/^(https?:|mailto:|data:)/.test(url)) continue;
    checked++;
    const [beforeHash, hash] = url.split('#');
    const p = beforeHash.split('?')[0];
    let target;
    if (p === '') target = file;
    else if (p.startsWith('/')) target = path.join(root, p);
    else target = path.join(path.dirname(file), p);
    if (fs.existsSync(target) && fs.statSync(target).isDirectory()) target = path.join(target, 'index.html');
    if (!fs.existsSync(target)) {
      console.error(`✗ ${rel}: ${url} → no existe`);
      errors++;
      continue;
    }
    if (hash && target.endsWith('.html') && !idsOf(target).has(hash)) {
      console.error(`✗ ${rel}: ${url} → falta id="${hash}"`);
      errors++;
    }
  }
}
console.log(`${checked} enlaces locales revisados, ${errors} errores.`);
process.exit(errors ? 1 : 0);
