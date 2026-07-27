#!/usr/bin/env node
// Transform committed unified GPU catalog descriptors into the cuVS SQL
// reference. The generator is deterministic and never contacts a server.
//
//   node website/scripts/gen-cuvs-docs.mjs
import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEBSITE = join(HERE, '..');
const SPEC_DIR = join(WEBSITE, 'gpu-function-specs', 'cuvs');
const DOCS_DIR = join(WEBSITE, 'docs', 'cuvs');
const OPERATIONS_DIR = join(DOCS_DIR, 'operations');
const REQUIRED_KEYS = [
  'function_name', 'provider', 'domain', 'available', 'unavailable_reason',
  'signature', 'summary', 'relation_roles_json', 'argument_descriptions_json',
  'options_schema_json', 'result_schemas_json',
  'validation_request_schema_json', 'examples_json', 'limitations_json',
];

function write(path, content) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, content.replace(/\n*$/, '\n'));
}

function parseJson(file, key, value) {
  try {
    return JSON.parse(value);
  } catch (error) {
    throw new Error(`${file}: ${key} is not valid JSON: ${error.message}`);
  }
}

function loadSpecs() {
  if (!existsSync(SPEC_DIR)) throw new Error(`missing ${SPEC_DIR}; run capture-cuvs-specs.sh first`);
  const names = new Set();
  const specs = readdirSync(SPEC_DIR)
    .filter(file => file.endsWith('.json'))
    .sort()
    .map(file => {
      const spec = JSON.parse(readFileSync(join(SPEC_DIR, file), 'utf8'));
      for (const key of REQUIRED_KEYS) {
        if (!(key in spec)) throw new Error(`${file}: missing unified descriptor key ${key}`);
      }
      if (file !== `${spec.function_name}.json`) throw new Error(`${file}: filename must equal ${spec.function_name}.json`);
      if (!/^cuvs_[a-z0-9_]+$/.test(spec.function_name)) throw new Error(`${file}: invalid canonical cuVS function name`);
      if (names.has(spec.function_name)) throw new Error(`${file}: duplicate function descriptor`);
      names.add(spec.function_name);
      if (spec.provider !== 'cuvs' || spec.domain !== 'vector') throw new Error(`${file}: expected a cuVS vector descriptor`);
      if (typeof spec.available !== 'boolean' || !(spec.unavailable_reason === null || typeof spec.unavailable_reason === 'string')) {
        throw new Error(`${file}: invalid availability fields`);
      }
      spec.relation_roles = parseJson(file, 'relation_roles_json', spec.relation_roles_json);
      spec.argument_descriptions = parseJson(file, 'argument_descriptions_json', spec.argument_descriptions_json);
      spec.options_schema = parseJson(file, 'options_schema_json', spec.options_schema_json);
      spec.result_schemas = parseJson(file, 'result_schemas_json', spec.result_schemas_json);
      spec.validation_request_schema = parseJson(file, 'validation_request_schema_json', spec.validation_request_schema_json);
      spec.examples = parseJson(file, 'examples_json', spec.examples_json);
      spec.limitations = parseJson(file, 'limitations_json', spec.limitations_json);
      if (!Array.isArray(spec.relation_roles) || spec.relation_roles.length === 0 || spec.relation_roles.some(role => !role?.name)) {
        throw new Error(`${file}: relation_roles_json must contain named roles`);
      }
      if (!spec.options_schema || typeof spec.options_schema !== 'object' || !spec.result_schemas || typeof spec.result_schemas !== 'object') {
        throw new Error(`${file}: descriptor schemas must be JSON objects`);
      }
      if (!Array.isArray(spec.examples) || !Array.isArray(spec.limitations)) throw new Error(`${file}: descriptor arrays have an invalid shape`);
      return spec;
    });
  if (specs.length === 0) throw new Error('no cuVS function descriptors found');
  return specs;
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
  if (value === '' || value == null) return '';
  const inner = String(value).replace(/\|/g, '\\|');
  const longest = (inner.match(/`+/g) || []).reduce((max, run) => Math.max(max, run.length), 0);
  const fence = '`'.repeat(longest + 1);
  return `${fence}${inner}${fence}`;
}

function prose(value) {
  return String(value ?? '')
    .replace(/[{}]/g, match => `\\${match}`)
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function table(headers, rows) {
  return [
    `| ${headers.join(' | ')} |`,
    `| ${headers.map(() => '---').join(' | ')} |`,
    ...rows.map(row => `| ${row.join(' | ')} |`),
  ].join('\n');
}

function relationNames(spec) {
  return spec.relation_roles.map(role => role.name);
}

function renderRelations(spec) {
  return table(['Role', 'Required', 'Reference', 'Description'], spec.relation_roles.map(role => [
    code(role.name), role.required === false ? 'no' : 'yes', code(role.reference?.type || 'table'), cell(role.description),
  ]));
}

function optionConstraints(option) {
  const constraints = [];
  if (option.enum) constraints.push(`one of ${option.enum.map(value => code(JSON.stringify(value))).join(', ')}`);
  if ('minimum' in option) constraints.push(`minimum ${option.minimum}`);
  if ('maximum' in option) constraints.push(`maximum ${option.maximum}`);
  if ('minLength' in option) constraints.push(`minimum string length ${option.minLength}`);
  if ('minItems' in option) constraints.push(`at least ${option.minItems} item${option.minItems === 1 ? '' : 's'}`);
  if (option.oneOf) constraints.push('one of the documented binding shapes');
  return constraints.join('; ');
}

function renderOptions(spec) {
  const required = new Set(spec.options_schema.required || []);
  const rows = Object.entries(spec.options_schema.properties || {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, option]) => [
      code(name),
      required.has(name) ? 'yes' : 'no',
      code(option.type ?? (option.oneOf ? 'object' : 'unknown')),
      Object.hasOwn(option, 'default') ? code(JSON.stringify(option.default)) : '',
      cell(optionConstraints(option)),
    ]);
  return rows.length === 0 ? '_This function has no options._' : table(['Option', 'Required', 'JSON type', 'Default', 'Constraints'], rows);
}

function renderOutputSchema(spec) {
  const fields = spec.result_schemas.fields;
  if (!Array.isArray(fields)) throw new Error(`${spec.function_name}: result_schemas_json must contain fields`);
  return table(['Column', 'Type', 'Nullable', 'Notes'], fields.map(field => [
    code(field.name), code(field.data_type), field.nullable ? 'yes' : 'no', cell(field.repeats ? `repeats: ${field.repeats}` : ''),
  ]));
}

function firstExample(spec) {
  const example = spec.examples.find(candidate => candidate?.validation?.call && typeof candidate.validation.call === 'object');
  if (!example) throw new Error(`${spec.function_name}: examples_json must contain a unified validation call`);
  return example.validation.call;
}

function relationSql(role) {
  const id = role === 'queries' ? 'query_id' : 'item_id';
  return `(SELECT ${id}, d0, d1 FROM ${role}_vectors)`;
}

function renderPage(spec) {
  const call = firstExample(spec);
  return [
    '---',
    `title: ${spec.function_name}`,
    `sidebar_label: ${spec.function_name}`,
    `description: ${JSON.stringify(spec.summary)}`,
    '---',
    '',
    '{/* GENERATED FROM the unified GPU function catalog by website/scripts/gen-cuvs-docs.mjs. Do not edit by hand. */}',
    '',
    `# ${spec.function_name}`,
    '',
    prose(spec.summary),
    '',
    `Provider: ${code(spec.provider)}. Domain: ${code(spec.domain)}.`,
    '',
    '## Signature',
    '',
    '```sql',
    spec.signature,
    '```',
    '',
    '## Execution call shape',
    '',
    'Each relation argument is a parenthesized `SELECT` subquery. The execution function preserves those subqueries as real planner children; a bare table identifier or quoted table-name string is intentionally rejected.',
    '',
    '```sql',
    `SELECT * FROM ${spec.function_name}(`,
    `  ${relationNames(spec).map(relationSql).join(',\n  ')},`,
    `  '${JSON.stringify(call.options)}'`,
    ');',
    '```',
    '',
    '## Validation relations',
    '',
    'For metadata validation, use registered named tables or views in the common envelope:',
    '',
    renderRelations(spec),
    '',
    '## Options JSON',
    '',
    renderOptions(spec),
    '',
    'Unknown option keys are rejected.',
    '',
    '## Output schema',
    '',
    renderOutputSchema(spec),
    '',
    '## Limitations & notes',
    '',
    spec.limitations.map(limitation => `- ${prose(limitation)}`).join('\n'),
    '',
    '## Validate before running',
    '',
    '```sql',
    `SELECT * FROM gpu_validate_call(\n  '${spec.function_name}',\n  '${JSON.stringify(call)}'\n);`,
    '```',
    '',
    'See [Discovery & Validation](../../cugraph/discovery-and-validation.mdx) for the common `gpu_validate_call` contract.',
  ].join('\n');
}

function renderCheatsheet(specs) {
  return [
    '{/* GENERATED FROM the unified GPU function catalog by website/scripts/gen-cuvs-docs.mjs. Do not edit by hand. */}',
    '',
    `The live catalog contains **${specs.length} SQL-visible cuVS functions**.`,
    '',
    table(['Function', 'Relation roles', 'Summary'], specs.map(spec => [
      `[${cell(spec.function_name)}](./operations/${spec.function_name}.mdx)`,
      relationNames(spec).map(code).join(', '),
      cell(spec.summary),
    ])),
  ].join('\n');
}

function main() {
  const specs = loadSpecs();
  write(join(OPERATIONS_DIR, '_category_.json'), JSON.stringify({
    label: 'Operations',
    position: 1,
    link: { type: 'generated-index', description: 'Generated reference for every SQL-visible cuVS operation.' },
  }, null, 2));
  for (const spec of specs) write(join(OPERATIONS_DIR, `${spec.function_name}.mdx`), renderPage(spec));
  write(join(DOCS_DIR, '_cheatsheet.mdx'), renderCheatsheet(specs));
  console.log(`Wrote ${specs.length} generated cuVS function pages and cheatsheet.`);
}

main();
