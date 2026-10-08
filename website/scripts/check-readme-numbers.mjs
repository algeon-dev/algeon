// Checks the numbers in the root README.md that come from demo captures.
//
// README.md is static Markdown and cannot import the capture JSONs the way the
// site pages do, so each such number is preceded by a marker:
//
//   <!-- capture:<file under static/data>:<dotted path>:<format> -->3.82
//
// Formats: fixedN (toFixed(N)) and int (grouped integer). The script fails
// when a marked number differs from the capture, so a recapture that changes
// a result also has to update the README.
import {readFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const website = dirname(dirname(fileURLToPath(import.meta.url)));
const readmePath = join(website, '..', 'README.md');
const readme = readFileSync(readmePath, 'utf8');

const MARKER = /<!-- capture:([^:>\s]+):([^:>\s]+):(fixed\d|int) -->\s*([\d.,]+)/g;

function lookup(data, path) {
  return path.split('.').reduce((value, key) => (value == null ? value : value[key]), data);
}

function format(value, spec) {
  if (spec === 'int') {
    return Math.round(value).toLocaleString('en-US');
  }
  return Number(value).toFixed(Number(spec.slice('fixed'.length)));
}

const captures = new Map();
const mismatches = [];
let count = 0;
for (const [, file, path, spec, written] of readme.matchAll(MARKER)) {
  count += 1;
  if (!captures.has(file)) {
    captures.set(file, JSON.parse(readFileSync(join(website, 'static', 'data', file), 'utf8')));
  }
  const value = lookup(captures.get(file), path);
  if (typeof value !== 'number') {
    mismatches.push(`${file}:${path} is not a number in the capture`);
    continue;
  }
  const expected = format(value, spec);
  if (expected !== written) {
    mismatches.push(`${file}:${path} is ${expected} in the capture, README says ${written}`);
  }
}

if (mismatches.length > 0) {
  console.error(`README.md capture numbers out of date:\n  ${mismatches.join('\n  ')}`);
  process.exit(1);
}
console.log(`README.md: ${count} capture number${count === 1 ? '' : 's'} match static/data.`);
