// Shared deterministic rendering primitives for the cuGraph and cuVS SQL
// references. Provider adapters own their relation syntax, option schemas, and
// presentation families; this module owns the common page contract.

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';

export const REQUIRED_DESCRIPTOR_KEYS = [
  'function_name', 'provider', 'available', 'unavailable_reason',
  'signature', 'summary', 'relation_roles_json', 'argument_descriptions_json',
  'options_schema_json', 'result_schemas_json',
  'validation_request_schema_json', 'examples_json', 'limitations_json',
];

export function write(path, content) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, content.replace(/\n*$/, '\n'));
}

export function parseJson(file, key, value) {
  try {
    return JSON.parse(value);
  } catch (error) {
    throw new Error(`${file}: ${key} is not valid JSON: ${error.message}`);
  }
}

export function loadProviderSpecs({ specDir, provider, namePattern, validate }) {
  if (!existsSync(specDir)) throw new Error(`missing ${specDir}; run the descriptor capture script first`);
  const names = new Set();
  const specs = readdirSync(specDir)
    .filter(file => file.endsWith('.json'))
    .sort()
    .map(file => {
      const spec = JSON.parse(readFileSync(join(specDir, file), 'utf8'));
      for (const key of REQUIRED_DESCRIPTOR_KEYS) {
        if (!(key in spec)) throw new Error(`${file}: missing unified descriptor key ${key}`);
      }
      if (file !== `${spec.function_name}.json`) throw new Error(`${file}: filename must equal ${spec.function_name}.json`);
      if (!namePattern.test(spec.function_name)) throw new Error(`${file}: invalid canonical ${provider} function name`);
      if (names.has(spec.function_name)) throw new Error(`${file}: duplicate function descriptor`);
      names.add(spec.function_name);
      if (spec.provider !== provider) throw new Error(`${file}: expected a ${provider} descriptor`);
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
      validate(file, spec);
      return spec;
    });
  if (specs.length === 0) throw new Error(`no ${provider} function descriptors found`);
  return specs.sort((left, right) => left.function_name.localeCompare(right.function_name));
}

export function cell(value) {
  return String(value ?? '')
    .replace(/\\/g, '\\\\')
    .replace(/\|/g, '\\|')
    .replace(/[{}]/g, match => `\\${match}`)
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\n/g, ' ');
}

export function code(value) {
  if (value === '' || value == null) return '';
  const inner = String(value).replace(/\|/g, '\\|');
  const longest = (inner.match(/`+/g) || []).reduce((max, run) => Math.max(max, run.length), 0);
  const fence = '`'.repeat(longest + 1);
  const pad = inner.startsWith('`') || inner.endsWith('`') ? ' ' : '';
  return `${fence}${pad}${inner}${pad}${fence}`;
}

export function prose(value) {
  return String(value ?? '')
    .replace(/[{}]/g, match => `\\${match}`)
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

export function table(headers, rows) {
  return [
    `| ${headers.join(' | ')} |`,
    `| ${headers.map(() => '---').join(' | ')} |`,
    ...rows.map(row => `| ${row.join(' | ')} |`),
  ].join('\n');
}

export function sqlBlock(sql) {
  if (typeof sql !== 'string' || sql.trim() === '') throw new Error('generated page has no Quickstart SQL');
  return ['```sql', sql, '```'].join('\n');
}

export function narrativeExampleImport(importPath) {
  return [`import Examples from '${importPath}';`, '', '<Examples />'].join('\n');
}

export function renderFunctionPage({
  generator,
  spec,
  title,
  reference,
  quickstart,
  inputs,
  argumentsAndOptions,
  output,
  examples,
  limits,
  discoveryLink,
}) {
  // Page order is fixed: purpose, minimal call, inputs, arguments and
  // options, output, optional long-form examples, limits.
  const parts = [
    '---',
    `title: ${title}`,
    `sidebar_label: ${title}`,
    `description: ${JSON.stringify(spec.summary)}`,
    '---',
    '',
    `{/* GENERATED FROM the unified GPU function catalog by ${generator}. Do not edit by hand. */}`,
    '',
    `# ${title}`,
    '',
    `SQL function: ${code(spec.function_name)}`,
  ];
  if (reference != null) parts.push('', reference);
  parts.push('', prose(spec.summary));
  const headings = ['Quickstart', 'Inputs', 'Arguments and options', 'Output'];
  parts.push(
    '',
    '## Quickstart',
    '',
    quickstartBlock(quickstart),
    '',
    '## Inputs',
    '',
    inputs,
    '',
    '## Arguments and options',
    '',
    argumentsAndOptions,
    '',
    '## Output',
    '',
    output,
  );
  if (examples) {
    parts.push('', '## Examples', '', examples);
    headings.push('Examples');
  }
  const limitsParts = [limits];
  if (discoveryLink) {
    limitsParts.push('', `To dry-run validate relation metadata, column types, and options without execution, see [\`gpu_validate_call\`](${discoveryLink}).`);
  }
  parts.push('', '## Limits', '', limitsParts.join('\n'));
  headings.push('Limits');
  const page = parts.join('\n');
  assertPageContract(spec.function_name, page, headings);
  return page;
}

function quickstartBlock({ prerequisite, sql }) {
  if (typeof prerequisite !== 'string' || prerequisite.trim() === '') throw new Error('generated page has no Quickstart prerequisite');
  return [prerequisite, '', sqlBlock(sql)].join('\n');
}

export function assertPageContract(functionName, page, headings) {
  let previous = -1;
  for (const heading of headings) {
    const index = page.indexOf(`## ${heading}`);
    if (index === -1) throw new Error(`${functionName}: generated page is missing common heading '${heading}'`);
    if (index <= previous) throw new Error(`${functionName}: common headings are out of order at '${heading}'`);
    previous = index;
  }
}
