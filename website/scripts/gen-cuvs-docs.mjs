#!/usr/bin/env node
// GENERATOR for the cuVS SQL API reference. It transforms only committed
// metadata captured by capture-cuvs-specs.sh; site builds never call a server.
//
//   node website/scripts/gen-cuvs-docs.mjs
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEBSITE = join(HERE, '..');
const SPEC_DIR = join(WEBSITE, 'cuvs-specs');
const DOCS_DIR = join(WEBSITE, 'docs', 'cuvs');
const OPERATIONS_DIR = join(DOCS_DIR, 'operations');
const REQUIRED_KEYS = [
  'function_name',
  'operation',
  'relation_roles_json',
  'relation_argument_syntax',
  'output_contract',
  'options_schema_json',
];

function write(path, content) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, content.replace(/\n*$/, '\n'));
}

function cell(value) {
  return String(value ?? '')
    .replace(/\\/g, '\\\\')
    .replace(/\|/g, '\\|')
    .replace(/[{}]/g, match => `\\${match}`)
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, ' ');
}

function code(value) {
  const inner = String(value ?? '').replace(/\|/g, '\\|');
  const longest = (inner.match(/`+/g) || []).reduce((max, run) => Math.max(max, run.length), 0);
  return `${'`'.repeat(longest + 1)}${inner}${'`'.repeat(longest + 1)}`;
}

function table(headers, rows) {
  return [
    `| ${headers.join(' | ')} |`,
    `| ${headers.map(() => '---').join(' | ')} |`,
    ...rows.map(row => `| ${row.join(' | ')} |`),
  ].join('\n');
}

function loadSpecs() {
  if (!existsSync(SPEC_DIR)) throw new Error(`missing ${SPEC_DIR}; run capture-cuvs-specs.sh first`);
  const names = new Set();
  return readdirSync(SPEC_DIR)
    .filter(file => file.endsWith('.json'))
    .sort()
    .map(file => {
      const spec = JSON.parse(readFileSync(join(SPEC_DIR, file), 'utf8'));
      for (const key of REQUIRED_KEYS) {
        if (!(key in spec)) throw new Error(`${file}: missing required metadata key ${key}`);
      }
      if (file !== `${spec.function_name}.json`) {
        throw new Error(`${file}: filename must equal function_name ${spec.function_name}.json`);
      }
      if (!/^cuvs_[a-z0-9_]+$/.test(spec.function_name)) {
        throw new Error(`${file}: invalid SQL function name ${spec.function_name}`);
      }
      if (names.has(spec.function_name)) throw new Error(`${file}: duplicate operation ${spec.function_name}`);
      names.add(spec.function_name);
      spec.relation_roles = JSON.parse(spec.relation_roles_json);
      spec.options_schema = JSON.parse(spec.options_schema_json);
      if (!Array.isArray(spec.relation_roles) || spec.relation_roles.length === 0) {
        throw new Error(`${file}: relation_roles_json must describe at least one relation`);
      }
      if (spec.relation_argument_syntax !== 'parenthesized_subquery') {
        throw new Error(`${file}: unsupported relation argument syntax ${spec.relation_argument_syntax}`);
      }
      if (spec.options_schema.type !== 'object' || spec.options_schema.additionalProperties !== false) {
        throw new Error(`${file}: options schema must be a closed JSON object`);
      }
      return spec;
    });
}

function outputRows(contract) {
  switch (contract) {
    case 'brute_force_knn':
      return [
        [code('query_ordinal'), code('UInt64'), 'Stable zero-based input query row ordinal.'],
        [code('query_id'), code('same type as queries.id'), 'Query identifier.'],
        [code('neighbor_ordinal'), code('UInt64'), 'Stable zero-based dataset row ordinal.'],
        [code('neighbor_id'), code('same type as dataset.id'), 'Neighbor identifier.'],
        [code('rank'), code('UInt32'), 'Zero-based nearest-neighbor rank.'],
        [code('distance'), code('Float32'), 'Metric result for the selected neighbor.'],
      ];
    case 'kmeans_assignments':
      return [
        [code('row_ordinal'), code('UInt64'), 'Stable zero-based input row ordinal.'],
        [code('id'), code('same type as input.id'), 'Input identifier.'],
        [code('cluster_id'), code('Int32'), 'Fitted-cluster assignment.'],
      ];
    case 'pca_components':
      return [
        [code('row_ordinal'), code('UInt64'), 'Stable zero-based input row ordinal.'],
        [code('id'), code('same type as input.id'), 'Input identifier.'],
        [code('pc_0 … pc_(n_components - 1)'), code('Float32'), 'One transformed component column for each requested component.'],
      ];
    default:
      throw new Error(`unknown output_contract ${contract}; extend outputRows so metadata is not silently dropped`);
  }
}

function optionConstraints(option) {
  const values = [];
  if (option.enum) values.push(`one of ${option.enum.map(value => code(JSON.stringify(value))).join(', ')}`);
  if ('minimum' in option) values.push(`minimum ${option.minimum}`);
  if ('maximum' in option) values.push(`maximum ${option.maximum}`);
  if ('minLength' in option) values.push(`minimum string length ${option.minLength}`);
  if ('minItems' in option) values.push(`at least ${option.minItems} item${option.minItems === 1 ? '' : 's'}`);
  if (option.oneOf) values.push('one of wide Float32 columns or a Float32 vector column');
  return values.join('; ');
}

function optionRows(spec) {
  const required = new Set(spec.options_schema.required || []);
  return Object.entries(spec.options_schema.properties || {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, option]) => [
      code(name),
      required.has(name) ? 'yes' : 'no',
      code(option.type ?? (option.oneOf ? 'relation binding' : 'unknown')),
      Object.hasOwn(option, 'default') ? code(JSON.stringify(option.default)) : '',
      cell(optionConstraints(option)),
    ]);
}

function exampleOptions(spec) {
  const options = {};
  for (const role of spec.relation_roles) {
    options[role] = { id: role === 'queries' ? 'query_id' : 'item_id', vector: { columns: ['d0', 'd1'] } };
  }
  for (const [name, option] of Object.entries(spec.options_schema.properties || {})) {
    if (name in options || !spec.options_schema.required?.includes(name)) continue;
    if (option.enum) options[name] = option.enum[0];
    else if (option.type === 'integer') options[name] = name === 'n_components' ? 2 : 1;
    else if (option.type === 'number') options[name] = option.minimum ?? 0;
    else if (option.type === 'boolean') options[name] = false;
    else throw new Error(`${spec.function_name}: cannot make a generated example for required option ${name}`);
  }
  return JSON.stringify(options);
}

function relationSql(spec) {
  return spec.relation_roles
    .map(role => `(SELECT ${role === 'queries' ? 'query_id' : 'item_id'}, d0, d1 FROM ${role}_vectors)`)
    .join(', ');
}

function renderPage(spec) {
  const options = exampleOptions(spec);
  return [
    '---',
    `title: ${spec.function_name}`,
    `sidebar_label: ${spec.function_name}`,
    `description: ${JSON.stringify(`Generated reference for ${spec.function_name}.`)}`,
    '---',
    '',
    '{/* GENERATED FROM cuVS SQL metadata by website/scripts/gen-cuvs-docs.mjs. Do not edit by hand. */}',
    '',
    `# ${spec.function_name}`,
    '',
    `Native operation: ${code(spec.operation)}. Output contract: ${code(spec.output_contract)}.`,
    '',
    '## Call shape',
    '',
    'Every relation argument is a parenthesized `SELECT` subquery. Nexus resolves the tables reached by those subqueries before logical planning and preserves the subqueries as real children. A bare relation identifier or quoted table-name string is intentionally rejected.',
    '',
    '```sql',
    `SELECT * FROM ${spec.function_name}(`,
    `  ${relationSql(spec)},`,
    `  '${options}'`,
    ');',
    '```',
    '',
    `Relation roles, in argument order: ${spec.relation_roles.map(code).join(', ')}.`,
    '',
    'A binding names a non-null `Int32`, `Int64`, `Utf8`, `LargeUtf8`, or `Utf8View` ID column and either non-null `Float32` columns or one non-null `FixedSizeList<Float32, D>`, `List<Float32>`, or `LargeList<Float32>` column. Variable-width lists are checked for a uniform dimension on the device.',
    '',
    '## Options JSON',
    '',
    table(['Option', 'Required', 'JSON type', 'Default', 'Constraints'], optionRows(spec)),
    '',
    'Unknown option keys are rejected. Use `cuvs_validate_call` to canonicalize an options JSON object before executing it.',
    '',
    '## Output schema',
    '',
    table(['Column', 'Type', 'Meaning'], outputRows(spec.output_contract)),
    '',
    'Ordering is deterministic by the declared ordinals and rank. Use `ORDER BY` when a different presentation order is required.',
  ].join('\n');
}

function renderCheatsheet(specs) {
  const rows = specs.map(spec => [
    `[${cell(spec.function_name)}](./operations/${spec.function_name}.mdx)`,
    code(spec.operation),
    spec.relation_roles.map(code).join(', '),
    code(spec.output_contract),
  ]);
  return [
    '{/* GENERATED FROM cuVS SQL metadata by website/scripts/gen-cuvs-docs.mjs. Do not edit by hand. */}',
    '',
    `The live registry exposes **${specs.length} SQL-visible cuVS operations**.`,
    '',
    table(['SQL function', 'Operation', 'Relation arguments', 'Output contract'], rows),
  ].join('\n');
}

function main() {
  const specs = loadSpecs();
  if (specs.length === 0) throw new Error('no cuVS operation specs found');
  write(join(OPERATIONS_DIR, '_category_.json'), JSON.stringify({
    label: 'Operations',
    position: 1,
    link: { type: 'generated-index', description: 'Generated reference for every SQL-visible cuVS operation.' },
  }, null, 2));
  for (const spec of specs) write(join(OPERATIONS_DIR, `${spec.function_name}.mdx`), renderPage(spec));
  write(join(DOCS_DIR, '_cheatsheet.mdx'), renderCheatsheet(specs));
  console.log(`Wrote ${specs.length} generated cuVS operation pages and cheatsheet.`);
}

main();
