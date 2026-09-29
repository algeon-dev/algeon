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

// A JSON-schema option without `default` must be supplied by the caller when
// required and is absent otherwise; an empty table cell hides that contract.
export function defaultCell(option) {
  return Object.hasOwn(option, 'default') ? code(JSON.stringify(option.default)) : 'No default';
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

export function formatFunctionQuickstartSql(sql) {
  const statement = sql.trim().replace(/;$/, '');
  const match = /^SELECT (.+?) FROM ([a-z_][a-z0-9_]*)\((.+)\)(?: (ORDER BY .+))?$/s.exec(statement);
  if (!match) throw new Error(`unsupported Quickstart SQL layout: ${sql}`);
  const [, projection, functionName, argumentsSql, orderBy] = match;
  // Split before named arguments, not between projected columns in relation subqueries.
  const args = argumentsSql.split(/, (?=[a-z_][a-z0-9_]* => )/);
  if (args.some(arg => !/^[a-z_][a-z0-9_]* => .+$/s.test(arg))) {
    throw new Error(`unsupported Quickstart arguments: ${sql}`);
  }
  return [
    `SELECT ${projection}`,
    `FROM ${functionName}(`,
    ...args.map((arg, index) => `  ${arg}${index < args.length - 1 ? ',' : ''}`),
    `)${orderBy ? `\n${orderBy}` : ''};`,
  ].join('\n');
}

export function narrativeExampleImport(importPath) {
  return [`import Examples from '${importPath}';`, '', '<Examples />'].join('\n');
}

export function renderFunctionPage({
  generator,
  spec,
  title,
  slug,
  sidebarPosition,
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
  ];
  if (slug) {
    parts.push(`slug: ${slug}`);
  }
  if (sidebarPosition !== undefined) {
    parts.push(`sidebar_position: ${sidebarPosition}`);
  }
  parts.push(
    '---',
    '',
    `{/* GENERATED FROM the unified GPU UDTF catalog by ${generator}. Do not edit by hand. */}`,
    '',
    `# ${title}`,
    '',
    `UDTF: ${code(spec.function_name)}`,
  );
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

export function densePresentation(spec) {
  const entry = spec.examples.find(candidate => candidate?.quickstart?.sql && candidate?.validation?.call);
  if (!entry) throw new Error(`${spec.function_name}: examples_json must contain quickstart SQL and a validation call`);
  const quickstart = entry.quickstart;
  if (typeof quickstart.display_name !== 'string' || quickstart.display_name.trim() === '') {
    throw new Error(`${spec.function_name}: quickstart metadata must contain display_name`);
  }
  if (typeof quickstart.sql !== 'string' || quickstart.sql.trim() === '') {
    throw new Error(`${spec.function_name}: quickstart metadata must contain executable SQL`);
  }
  return entry;
}

const RENDERED_OPTION_KEYS = new Set([
  'type', 'default', 'enum', 'minimum', 'exclusiveMinimum', 'maximum', 'minLength', 'maxLength', 'minItems',
  'oneOf', 'description', 'additionalProperties', 'properties', 'required',
]);

export function assertRenderableOptions(spec) {
  for (const [name, option] of Object.entries(spec.options_schema.properties || {})) {
    for (const key of Object.keys(option)) {
      if (!RENDERED_OPTION_KEYS.has(key)) throw new Error(`${spec.function_name}: option ${name} has unrendered descriptor key ${key}`);
    }
    if (typeof option.description !== 'string' || option.description.trim() === '') {
      throw new Error(`${spec.function_name}: option ${name} has no descriptor description`);
    }
  }
}

export function assertRenderableOutputFields(spec) {
  for (const field of spec.result_schemas.fields) {
    for (const key of Object.keys(field)) {
      if (!['name', 'data_type', 'nullable', 'repeats', 'description'].includes(key)) {
        throw new Error(`${spec.function_name}: result field ${field.name || '<unnamed>'} has unrendered descriptor key ${key}`);
      }
    }
    if (typeof field.name !== 'string' || typeof field.data_type !== 'string' || typeof field.description !== 'string') {
      throw new Error(`${spec.function_name}: result fields require name, data_type, and description`);
    }
  }
}

function optionConstraints(option) {
  const constraints = [];
  if (option.enum) constraints.push(`one of ${option.enum.map(value => code(JSON.stringify(value))).join(', ')}`);
  if ('minimum' in option) constraints.push(`minimum ${option.minimum}`);
  if ('exclusiveMinimum' in option) constraints.push(`greater than ${option.exclusiveMinimum}`);
  if ('maximum' in option) constraints.push(`maximum ${option.maximum}`);
  if ('minLength' in option) constraints.push(`minimum string length ${option.minLength}`);
  if ('maxLength' in option) constraints.push(`maximum string length ${option.maxLength}`);
  if ('minItems' in option) constraints.push(`at least ${option.minItems} item${option.minItems === 1 ? '' : 's'}`);
  if (option.oneOf) constraints.push('one of the documented vector-binding shapes');
  return constraints.join('; ');
}

function optionType(option) {
  if (option.oneOf) return 'object (binding)';
  return option.type || 'unknown';
}

export function renderDenseArguments(spec, bindingShapes) {
  const argumentsTable = spec.argument_descriptions.length === 0
    ? '_This UDTF has no scalar SQL arguments beyond its relation arguments._'
    : table(['Argument', 'Type', 'Required', 'Description'], spec.argument_descriptions.map(argument => [
      code(argument.name),
      code(argument.type || sqlArgumentType(argument)),
      argument.required ? 'yes' : 'no',
      cell(argument.description),
    ]));
  const required = new Set(spec.options_schema.required || []);
  const rows = Object.entries(spec.options_schema.properties || {})
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, option]) => [
      code(name),
      required.has(name) ? 'yes' : 'no',
      code(optionType(option)),
      defaultCell(option),
      cell(optionConstraints(option)),
      cell(option.description),
    ]);
  return [
    '### Scalar SQL arguments',
    '',
    argumentsTable,
    '',
    '### SQL value argument schemas',
    '',
    rows.length === 0 ? '_This UDTF has no options._' : table(['Argument', 'Required', 'Literal shape', 'Default', 'Constraints', 'Description'], rows),
    '',
    '### Vector binding shapes',
    '',
    bindingShapes,
  ].join('\n');
}

function sqlArgumentType(argument) {
  const schema = argument.schema || {};
  if (schema.enum) return `enum (${schema.enum.map(value => JSON.stringify(value)).join(', ')})`;
  if (schema.type) return schema.type;
  return schema.oneOf ? 'object' : 'literal';
}

// A repeated field's descriptor name carries an `<index>` placeholder;
// its physical column count comes from the named literal option.
function repeatedFieldSentence(field) {
  const prefix = field.name.replace(/<index>$/, '');
  if (prefix === field.name) throw new Error(`repeated result field ${field.name} must end with <index>`);
  return `The output contains ${code(field.repeats)} such columns, ${code(`${prefix}0`)} through ${code(`${prefix}<${field.repeats} - 1>`)}.`;
}

export function renderDenseOutputSchema(spec) {
  return [
    table(['Column', 'Type', 'Nullable', 'Description'], spec.result_schemas.fields.map(field => [
      code(field.name),
      code(field.data_type),
      field.nullable ? 'yes' : 'no',
      cell(field.repeats ? `${field.description} ${repeatedFieldSentence(field)}` : field.description),
    ])),
    '',
    '**Concrete schemas are call-specific.** Run [`gpu_validate_call`](../../reference/gpu-udtf-catalog.mdx) against registered relations to inspect the output schema after the actual ID types and literal options are validated.',
  ].join('\n');
}

export function denseQuickstartPrerequisite(spec, entry) {
  const relations = Object.entries(entry.validation.call.relations || {});
  if (relations.length === 0) throw new Error(`${spec.function_name}: validation call names no relations`);
  const named = relations.map(([role, relation]) => `${code(relation.table)} (the ${code(role)} role)`);
  const passed = named.length > 1 ? 'each passed' : 'passed';
  return `The call below expects the registered relation${named.length > 1 ? 's' : ''} ${named.join(' and ')}, ${passed} as a parenthesized ${code('SELECT')} subquery. Substitute your own relations and column names.`;
}
