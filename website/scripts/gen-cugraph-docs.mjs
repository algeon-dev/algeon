#!/usr/bin/env node
// GENERATOR for the cuGraph SQL API docs. Pure transform: reads committed
// verbose specs from website/cugraph-specs/*.json and writes MDX under
// website/docs/cugraph/. Deterministic, no network, no server.
//
//   node website/scripts/gen-cugraph-docs.mjs
//
// Re-run after capture-cugraph-specs.sh refreshes the specs. Never writes into
// docs/cugraph/examples/ (hand-owned) or the hand-written cross-cutting pages.
import { readFileSync, readdirSync, writeFileSync, mkdirSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEBSITE = join(HERE, '..');
const SPEC_DIR = join(WEBSITE, 'cugraph-specs');
const DOCS_DIR = join(WEBSITE, 'docs', 'cugraph');
// Algorithm reference pages live one layer below the cuGraph guides
// (overview/connecting/discovery/graph-options), under cugraph/algorithms/.
const ALGORITHMS_DIR = join(DOCS_DIR, 'algorithms');
const EXPECTED_SCHEMA_VERSION = 11;
const REQUIRED_KEYS = [
  'function_name', 'algorithm', 'source_vertex_policy', 'weight_col_policy',
  'output_schema_kind', 'signature', 'allowed_arg_counts_json', 'summary',
  'primary_example_sql', 'result_schemas_json', 'options_schema_json',
  'argument_descriptions_json', 'result_field_descriptions_json',
  'limitations_json', 'examples_json', 'metadata_schema_version',
];

// family slug -> { label, position, functions[] } ; position drives sidebar order
const FAMILIES = {
  centrality:   { label: 'Centrality', position: 2, functions: ['cugraph_pagerank','cugraph_personalized_pagerank','cugraph_eigenvector_centrality','cugraph_katz_centrality','cugraph_hits','cugraph_betweenness_centrality','cugraph_edge_betweenness_centrality'] },
  community:    { label: 'Community Detection', position: 3, functions: ['cugraph_louvain','cugraph_leiden','cugraph_ecg','cugraph_spectral_modularity_maximization','cugraph_balanced_cut_clustering'] },
  traversal:    { label: 'Traversal & Paths', position: 4, functions: ['cugraph_bfs','cugraph_sssp'] },
  connectivity: { label: 'Connectivity', position: 5, functions: ['cugraph_weakly_connected_components','cugraph_strongly_connected_components','cugraph_minimum_spanning_tree'] },
  structure:    { label: 'Structure', position: 6, functions: ['cugraph_degrees_all','cugraph_in_degrees_all','cugraph_out_degrees_all','cugraph_triangle_count_all','cugraph_core_number','cugraph_k_core'] },
  layout:       { label: 'Layout', position: 7, functions: ['cugraph_force_atlas2'] },
};

function familyOf(fn) {
  for (const [slug, fam] of Object.entries(FAMILIES)) if (fam.functions.includes(fn)) return slug;
  throw new Error(`No family for ${fn} — update FAMILIES in gen-cugraph-docs.mjs`);
}

function loadSpecs() {
  const files = readdirSync(SPEC_DIR).filter(f => f.endsWith('.json')).sort();
  const specs = files.map(f => {
    const spec = JSON.parse(readFileSync(join(SPEC_DIR, f), 'utf8'));
    for (const k of REQUIRED_KEYS) {
      if (!(k in spec)) throw new Error(`${f}: missing required key "${k}" — metadata contract changed`);
    }
    if (spec.metadata_schema_version !== EXPECTED_SCHEMA_VERSION) {
      throw new Error(`${f}: metadata_schema_version ${spec.metadata_schema_version} != expected ${EXPECTED_SCHEMA_VERSION}. Review the spec, update EXPECTED_SCHEMA_VERSION and the renderers, then regenerate.`);
    }
    if (f !== `${spec.function_name}.json`) {
      throw new Error(`${f}: filename does not match function_name "${spec.function_name}" — a stale spec was left behind; re-run capture-cugraph-specs.sh`);
    }
    familyOf(spec.function_name); // throws if unclassified
    return spec;
  });
  // no duplicate function names, and every classified function has a spec
  const have = new Set();
  for (const s of specs) {
    if (have.has(s.function_name)) throw new Error(`duplicate spec for ${s.function_name}`);
    have.add(s.function_name);
  }
  for (const fam of Object.values(FAMILIES)) {
    for (const fn of fam.functions) if (!have.has(fn)) throw new Error(`FAMILIES lists ${fn} but no spec file found`);
  }
  return specs.sort((a, b) => a.function_name.localeCompare(b.function_name));
}

function write(path, content) {
  mkdirSync(dirname(path), { recursive: true });
  // normalize to a single trailing newline for deterministic diffs
  writeFileSync(path, content.replace(/\n*$/, '\n'));
}

function writeAlgorithmsCategory() {
  // The algorithms/ landing groups all per-UDF reference pages below the guides.
  const body = JSON.stringify({
    label: 'Algorithms',
    position: 5,
    link: { type: 'generated-index', description: 'Per-function reference for every cuGraph SQL algorithm, grouped by family.' },
  }, null, 2);
  write(join(ALGORITHMS_DIR, '_category_.json'), body);
}

function writeFamilyCategories() {
  for (const [slug, fam] of Object.entries(FAMILIES)) {
    const body = JSON.stringify({ label: fam.label, position: fam.position }, null, 2);
    write(join(ALGORITHMS_DIR, slug, '_category_.json'), body);
  }
}

// Human display name for a function (the spec's `algorithm` field), used as the
// page title and sidebar label — e.g. cugraph_bfs -> "BFS", cugraph_ecg -> "ECG".
const DISPLAY_NAME = new Map();

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

// MDX-safe inline text for table cells / prose: escape pipes (cells), braces
// (JSX expressions), and angle brackets so types like List<Int64> and the
// dtype union Int64|Utf8 render literally, and collapse newlines.
function cell(s) {
  if (s === null || s === undefined) return '';
  return String(s)
    .replace(/\\/g, '\\\\')
    .replace(/\|/g, '\\|')
    .replace(/[{}]/g, m => '\\' + m)
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, ' ');
}

// Inline code span. Uses a backtick fence longer than any run of backticks in
// the payload (the CommonMark rule), with surrounding spaces when the content
// starts/ends with a backtick, so embedded backticks render literally. Pipes
// are escaped because code spans still need escaping inside Markdown table cells.
function code(s) {
  if (s === '' || s == null) return '';
  const inner = String(s).replace(/\|/g, '\\|');
  const longest = (inner.match(/`+/g) || []).reduce((m, r) => Math.max(m, r.length), 0);
  const fence = '`'.repeat(longest + 1);
  const pad = inner.startsWith('`') || inner.endsWith('`') ? ' ' : '';
  return `${fence}${pad}${inner}${pad}${fence}`;
}

// MDX-safe inline prose (NOT inside a table cell): escape JSX braces and angle
// brackets so stray { } or < > can't start a JSX expression/element, but keep
// pipes and newlines intact for readability.
function prose(s) {
  if (s === null || s === undefined) return '';
  return String(s)
    .replace(/[{}]/g, m => '\\' + m)
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function table(headers, rows) {
  const head = `| ${headers.join(' | ')} |`;
  const sep = `| ${headers.map(() => '---').join(' | ')} |`;
  const body = rows.map(r => `| ${r.join(' | ')} |`).join('\n');
  return [head, sep, body].join('\n');
}

// Keys handled directly by the table columns (Option/Type/Default/Description)
// rather than the Constraints column.
const OPTION_COLUMN_KEYS = new Set(['type', 'default', 'description', 'scope']);

// Every option key the Constraints renderer knows how to surface. Any key in a
// spec that is NOT here (and not an OPTION_COLUMN_KEYS) makes the generator fail
// loudly — so new metadata can never be silently dropped from the docs.
const OPTION_CONSTRAINT_KEYS = new Set([
  'allowed_values', 'min', 'max', 'exclusive_min', 'required', 'required_when',
  'required_with', 'mutually_exclusive_with', 'valid_when', 'defaults_to',
  'column_of', 'expected_type_ref', 'example', 'side_input',
]);

// Constraint summary for an option object. Renders every known constraint key
// and throws on any unhandled key so the metadata contract stays complete.
function optionConstraints(name, o) {
  for (const k of Object.keys(o)) {
    if (!OPTION_COLUMN_KEYS.has(k) && !OPTION_CONSTRAINT_KEYS.has(k)) {
      throw new Error(`option "${name}" has unhandled key "${k}" — extend OPTION_CONSTRAINT_KEYS / optionConstraints() in gen-cugraph-docs.mjs so it is not dropped from the docs`);
    }
  }
  const parts = [];
  if (o.required === true) parts.push('required');
  if (o.allowed_values) parts.push(`one of ${o.allowed_values.map(v => code(JSON.stringify(v))).join(', ')}`);
  if ('min' in o) parts.push(`min ${o.min}`);
  if ('max' in o) parts.push(`max ${o.max}`);
  if ('exclusive_min' in o) parts.push(`> ${o.exclusive_min}`);
  if (o.required_when) parts.push(`required when ${o.required_when}`);
  if (o.required_with) parts.push(`required with ${code(o.required_with)}`);
  if (o.mutually_exclusive_with) parts.push(`mutually exclusive with ${code(o.mutually_exclusive_with)}`);
  if (o.valid_when) parts.push(`valid when ${o.valid_when}`);
  if (o.defaults_to) parts.push(`defaults to ${o.defaults_to}`);
  if (o.column_of) parts.push(`column of ${code(o.column_of)}`);
  if (o.expected_type_ref) parts.push(`type ref ${code(o.expected_type_ref)}`);
  if ('example' in o) parts.push(`example ${code(JSON.stringify(o.example))}`);
  if (o.side_input) {
    const role = o.side_input.role;
    const cols = o.side_input.column_option ? [o.side_input.column_option] : (o.side_input.column_options || []);
    parts.push(`side input (${role}${cols.length ? `, cols: ${cols.map(code).join(', ')}` : ''})`);
  }
  return parts.map(p => (typeof p === 'string' ? cell(p) : p)).join('; ');
}

function optionDescription(name, o) {
  if (name === 'construction_policy' && o.scope === 'graph') {
    return 'Edge-list construction semantics used before calling libcugraph.';
  }
  return o.description || '';
}

function optionComplexity(spec) {
  if (spec.function_name === 'cugraph_bfs') return 'rich';
  const opts = JSON.parse(spec.options_schema_json).algorithm_options || {};
  const n = Object.keys(opts).length;
  if (n === 0) return 'none';
  if (n <= 4) return 'few';
  return 'many';
}

const BANNER = (fn) => `{/* GENERATED FROM cuGraph SQL metadata by website/scripts/gen-cugraph-docs.mjs. Do not edit by hand; edit examples in website/docs/cugraph/_examples/${fn}.mdx and re-run the generator. */}`;

function renderArguments(spec) {
  const args = JSON.parse(spec.argument_descriptions_json);
  const rows = args.map(a => {
    const notes = [
      a.note,
      a.semantic_effect ? `semantic effect: ${a.semantic_effect}` : '',
      a.selector_policy ? `selector: ${a.selector_policy}` : '',
    ].filter(Boolean).join('; ');
    return [code(a.name), code(a.type), a.required ? 'yes' : 'no', a.default != null ? code(a.default) : '', cell(notes)];
  });
  return table(['Argument', 'Type', 'Required', 'Default', 'Notes'], rows);
}

function optionTable(opts) {
  const rows = Object.entries(opts).sort(([a], [b]) => a.localeCompare(b)).map(([k, v]) => [
    code(k),
    code(v.type),
    v.default === null ? code('null') : (v.default !== undefined ? code(JSON.stringify(v.default)) : ''),
    optionConstraints(k, v),
    cell(optionDescription(k, v)),
  ]);
  return table(['Option', 'Type', 'Default', 'Constraints', 'Description'], rows);
}

function renderOptions(spec) {
  const o = JSON.parse(spec.options_schema_json);
  const algo = o.algorithm_options || {};
  let out = '';
  if (Object.keys(algo).length === 0) {
    out += '_This algorithm has no algorithm-specific options._\n';
  } else {
    out += optionTable(algo);
  }
  // Render this function's actual graph_options so per-function defaults (e.g.
  // ForceAtlas2 directed=false) are visible, not hidden behind a generic link.
  const graph = o.graph_options || {};
  if (Object.keys(graph).length > 0) {
    out += `\n\n### Graph construction options\n\nShared by all cuGraph functions, shown here with this function's defaults. The \`construction_policy\` option controls whether Nexus requests Python cuGraph-compatible edge normalization or bypasses it for raw libcugraph-style construction; see [graph construction options](../../graph-options.mdx) for the full policy guide.\n\n`;
    out += optionTable(graph);
    out += '\n';
  }
  return out;
}

function renderSchemas(spec) {
  const rs = JSON.parse(spec.result_schemas_json);
  const fieldDescs = Object.fromEntries(JSON.parse(spec.result_field_descriptions_json).map(d => [d.name, d.description]));
  const def = rs.default_schema;
  const names = Object.keys(rs.schemas).sort((a, b) => (a === def ? -1 : b === def ? 1 : a.localeCompare(b)));
  const multi = names.length > 1;
  let out = '';
  for (const mode of names) {
    if (multi) out += `\n#### \`${mode}\`${mode === def ? ' (default)' : ''}\n\n`;
    const rows = rs.schemas[mode].fields.map(f => [code(f.name), code(f.data_type), f.nullable ? 'yes' : 'no', cell(fieldDescs[f.name] || '')]);
    out += table(['Column', 'Type', 'Nullable', 'Description'], rows) + '\n';
  }
  out += `\n:::note\nThese are the generic registry schemas. Run [\`cugraph_validate_call\`](../../discovery-and-validation.mdx#cugraph_validate_call) for the concrete, table-specific output schema of a particular call.\n:::\n`;
  return out;
}

function renderPage(spec) {
  const fn = spec.function_name;
  const argCounts = JSON.parse(spec.allowed_arg_counts_json).join(', ');
  const validateCall = `SELECT * FROM cugraph_validate_call(\n  '${fn}',\n  'your_edges_table',\n  '{"src_col":"src","dst_col":"dst"}'\n);`;
  return [
    '---',
    `title: ${spec.algorithm}`,
    `sidebar_label: ${spec.algorithm}`,
    `description: ${JSON.stringify(spec.summary)}`,
    '---',
    '',
    BANNER(fn),
    '',
    `# ${spec.algorithm}`,
    '',
    `SQL function: \`${fn}\``,
    '',
    prose(spec.summary),
    '',
    '## Signature',
    '',
    '```sql',
    spec.signature,
    '```',
    '',
    `Allowed argument counts: ${argCounts}.`,
    '',
    '## Quickstart',
    '',
    '```sql',
    spec.primary_example_sql,
    '```',
    '',
    '## Positional arguments',
    '',
    renderArguments(spec),
    '',
    '## JSON options',
    '',
    renderOptions(spec),
    '',
    '## Output schema',
    '',
    renderSchemas(spec),
    '',
    '## Examples',
    '',
    `import Examples from '../../_examples/${fn}.mdx';`,
    '',
    '<Examples />',
    '',
    '## Limitations & notes',
    '',
    JSON.parse(spec.limitations_json).map(l => `- ${prose(l)}`).join('\n'),
    '',
    '## Validate before running',
    '',
    'Always dry-run a call before executing it. Validation checks the function, table, columns, dtypes, and options without touching the GPU:',
    '',
    '```sql',
    validateCall,
    '```',
    '',
    `See [Discovery & validation](../../discovery-and-validation.mdx) for the full \`cugraph_validate_call\` contract.`,
  ].join('\n');
}

function renderCheatsheet(specs) {
  const rows = specs.map(s => {
    const fn = s.function_name;
    const slug = familyOf(fn);
    const fam = FAMILIES[slug].label;
    // Links resolve relative to overview.mdx (which imports this partial), so
    // they point into the algorithms/ subtree.
    const link = `[${cell(s.algorithm)}](./algorithms/${slug}/${fn}.mdx)`;
    return [link, code(fn), cell(fam), code(s.source_vertex_policy), code(s.weight_col_policy), code(s.output_schema_kind), optionComplexity(s), cell(s.summary)];
  });
  return [
    '{/* GENERATED FROM cuGraph SQL metadata by website/scripts/gen-cugraph-docs.mjs. Do not edit by hand. */}',
    '',
    table(['Algorithm', 'SQL function', 'Family', 'Source policy', 'Weight policy', 'Output kind', 'Options', 'Summary'], rows),
  ].join('\n');
}

function seedExample(spec) {
  const fn = spec.function_name;
  // Underscore-prefixed dir keeps these out of the docs route/sidebar; they are
  // imported as partials into each algorithm page, not standalone docs.
  const path = join(DOCS_DIR, '_examples', `${fn}.mdx`);
  if (existsSync(path)) return false; // never clobber hand-written examples
  const body = [
    `{/* Hand-owned examples for ${fn}. NOT generated — safe to edit. */}`,
    `{/* Add dataset-specific runnable examples here against your own tables. */}`,
    '',
    'Canonical call (generic table/column names):',
    '',
    '```sql',
    spec.primary_example_sql,
    '```',
    '',
    ':::info[Dataset-specific examples]',
    'TODO: add runnable examples against a concrete dataset (edge table, real',
    'vertex ids, expected output). These are intentionally left for a follow-up.',
    ':::',
  ].join('\n');
  write(path, body);
  return true;
}

function main() {
  const specs = loadSpecs();
  for (const spec of specs) DISPLAY_NAME.set(spec.function_name, spec.algorithm);
  writeAlgorithmsCategory();
  writeFamilyCategories();
  let seeded = 0;
  for (const spec of specs) {
    const fam = familyOf(spec.function_name);
    write(join(ALGORITHMS_DIR, fam, `${spec.function_name}.mdx`), renderPage(spec));
    if (seedExample(spec)) seeded++;
  }
  write(join(DOCS_DIR, '_cheatsheet.mdx'), renderCheatsheet(specs));
  console.log(`Wrote ${specs.length} pages, ${Object.keys(FAMILIES).length} families under algorithms/, cheatsheet, and ${seeded} example seeds.`);
}

main();
