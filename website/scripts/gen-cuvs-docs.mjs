#!/usr/bin/env node
// Transform committed unified GPU catalog descriptors into the cuVS SQL
// reference. The generator is deterministic and never contacts a server.
//
//   node website/scripts/gen-cuvs-docs.mjs
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  cell,
  code,
  loadProviderSpecs,
  prose,
  renderFunctionPage,
  table,
  write,
} from './gpu-docs-common.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEBSITE = join(HERE, '..');
const SPEC_DIR = join(WEBSITE, 'gpu-function-specs', 'cuvs');
const DOCS_DIR = join(WEBSITE, 'docs', 'cuvs');
const OPERATIONS_DIR = join(DOCS_DIR, 'operations');

// These presentation families are documentation navigation only. They do not
// alter SQL names or introduce an execution API family.
const FAMILIES = {
  vector_search: { label: 'Vector Search', functions: ['cuvs_brute_force_knn', 'cuvs_ivf_flat', 'cuvs_ivf_pq', 'cuvs_cagra'] },
  distances_and_selection: { label: 'Distances and Selection', functions: ['cuvs_pairwise_distance', 'cuvs_top_k'] },
  clustering: { label: 'Clustering', functions: ['cuvs_kmeans'] },
  dimensionality_reduction: { label: 'Dimensionality Reduction', functions: ['cuvs_pca'] },
};

function familyOf(functionName) {
  for (const [slug, family] of Object.entries(FAMILIES)) {
    if (family.functions.includes(functionName)) return slug;
  }
  throw new Error(`No documentation family for ${functionName}; classify it before generating docs`);
}

function loadSpecs() {
  return loadProviderSpecs({
    specDir: SPEC_DIR,
    provider: 'cuvs',
    namePattern: /^cuvs_[a-z0-9_]+$/,
    validate(file, spec) {
      if (!Array.isArray(spec.relation_roles) || spec.relation_roles.length === 0 || spec.relation_roles.some(role => !role?.name)) {
        throw new Error(`${file}: relation_roles_json must contain named roles`);
      }
      const elements = spec.relation_roles[0]?.fields?.features?.element_types;
      if (!Array.isArray(elements) || elements.length === 0 || elements.some(element => typeof element !== 'string') ||
          spec.relation_roles.some(role => JSON.stringify(role.fields?.features?.element_types) !== JSON.stringify(elements))) {
        throw new Error(`${file}: vector element types must be identical across relation roles`);
      }
      if (!Array.isArray(spec.argument_descriptions) || !Array.isArray(spec.examples) || !Array.isArray(spec.limitations)) {
        throw new Error(`${file}: descriptor arrays have an invalid shape`);
      }
      if (!spec.options_schema || typeof spec.options_schema !== 'object' || !spec.result_schemas || typeof spec.result_schemas !== 'object') {
        throw new Error(`${file}: descriptor schemas must be JSON objects`);
      }
      if (!Array.isArray(spec.result_schemas.fields)) throw new Error(`${file}: result_schemas_json must contain fields`);
      spec.family = familyOf(spec.function_name);
      presentation(spec);
      assertRenderableOptions(spec);
      assertRenderableOutputFields(spec);
    },
  });
}

function presentation(spec) {
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
  'type', 'default', 'enum', 'minimum', 'exclusiveMinimum', 'maximum', 'minLength', 'minItems',
  'oneOf', 'description', 'additionalProperties', 'properties', 'required',
]);

function assertRenderableOptions(spec) {
  for (const [name, option] of Object.entries(spec.options_schema.properties || {})) {
    for (const key of Object.keys(option)) {
      if (!RENDERED_OPTION_KEYS.has(key)) throw new Error(`${spec.function_name}: option ${name} has unrendered descriptor key ${key}`);
    }
    if (typeof option.description !== 'string' || option.description.trim() === '') {
      throw new Error(`${spec.function_name}: option ${name} has no descriptor description`);
    }
  }
}

function assertRenderableOutputFields(spec) {
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

function vectorElementTypes(spec) {
  return spec.relation_roles[0].fields.features.element_types;
}

function renderInputs(spec) {
  return [
    'Each relation argument is a parenthesized `SELECT` subquery that the planner keeps as a real child; metadata validation resolves a registered table or view for the same role instead. See [Vector Inputs](../vector-inputs.mdx) for the relation identity rules and the ID, dense-vector type, null, finite-value, and runtime-dimension contract.',
    '',
    table(['Role', 'Required', 'Validation reference', 'Description'], spec.relation_roles.map(role => [
      code(role.name), role.required === false ? 'no' : 'yes', code(role.reference?.type || 'table'), cell(role.description),
    ])),
    '',
    '### Vector element types',
    '',
    table(['Supported element type'], vectorElementTypes(spec).map(element => [code(element)])),
  ].join('\n');
}

function optionConstraints(option) {
  const constraints = [];
  if (option.enum) constraints.push(`one of ${option.enum.map(value => code(JSON.stringify(value))).join(', ')}`);
  if ('minimum' in option) constraints.push(`minimum ${option.minimum}`);
  if ('exclusiveMinimum' in option) constraints.push(`greater than ${option.exclusiveMinimum}`);
  if ('maximum' in option) constraints.push(`maximum ${option.maximum}`);
  if ('minLength' in option) constraints.push(`minimum string length ${option.minLength}`);
  if ('minItems' in option) constraints.push(`at least ${option.minItems} item${option.minItems === 1 ? '' : 's'}`);
  if (option.oneOf) constraints.push('one of the documented vector-binding shapes');
  return constraints.join('; ');
}

function optionType(option) {
  if (option.oneOf) return 'object (binding)';
  return option.type || 'unknown';
}

function renderArguments(spec) {
  const argumentsTable = spec.argument_descriptions.length === 0
    ? '_This function has no scalar SQL arguments beyond its relation arguments._'
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
      Object.hasOwn(option, 'default') ? code(JSON.stringify(option.default)) : '',
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
    rows.length === 0 ? '_This function has no options._' : table(['Argument', 'Required', 'Literal shape', 'Default', 'Constraints', 'Description'], rows),
    '',
    '### Vector binding shapes',
    '',
    renderVectorBindingShapes(spec),
  ].join('\n');
}

function sqlArgumentType(argument) {
  const schema = argument.schema || {};
  if (schema.enum) return `enum (${schema.enum.map(value => JSON.stringify(value)).join(', ')})`;
  if (schema.type) return schema.type;
  return schema.oneOf ? 'object' : 'literal';
}

function renderVectorBindingShapes(spec) {
  const bindingOption = Object.values(spec.options_schema.properties || {}).find(option => option?.properties?.vector?.oneOf);
  if (!bindingOption) {
    return [
      `Each relation subquery must project a non-null ${code('id')} field followed by either one or more non-null feature fields of a supported element type (${vectorElementTypes(spec).map(code).join(', ')}) or one non-null list vector field named ${code('vector')}.`,
      '',
      'For wide vectors, the projection order defines the feature dimensions. A list vector relation must contain no feature field beside `id` and `vector`.',
    ].join('\n');
  }
  const binding = bindingOption.properties;
  const alternatives = binding.vector.oneOf;
  const columns = alternatives.find(shape => shape?.properties?.columns)?.properties?.columns;
  const column = alternatives.find(shape => shape?.properties?.column)?.properties?.column;
  if (!binding.id?.description || !binding.vector?.description || !columns?.description || !column?.description) {
    throw new Error(`${spec.function_name}: vector binding descriptions are incomplete`);
  }
  return [
    `${code('id')}: ${prose(binding.id.description)}`,
    '',
    table(['Shape', 'JSON', 'Contract'], [
      [
        'Wide feature columns',
        code('{"vector":{"columns":["d0","d1"]}}'),
        cell(columns.description),
      ],
      [
        'List column',
        code('{"vector":{"column":"embedding"}}'),
        cell(column.description),
      ],
    ]),
    '',
    prose(binding.vector.description),
  ].join('\n');
}

// A repeated field's descriptor name carries an `<index>` placeholder; the
// physical column count equals the value of the literal option it names.
function repeatedFieldSentence(field) {
  const prefix = field.name.replace(/<index>$/, '');
  if (prefix === field.name) throw new Error(`repeated result field ${field.name} must end with <index>`);
  return `The output contains ${code(field.repeats)} such columns, ${code(`${prefix}0`)} through ${code(`${prefix}<${field.repeats} - 1>`)}.`;
}

function renderOutputSchema(spec) {
  return [
    table(['Column', 'Type', 'Nullable', 'Description'], spec.result_schemas.fields.map(field => [
      code(field.name),
      code(field.data_type),
      field.nullable ? 'yes' : 'no',
      cell(field.repeats ? `${field.description} ${repeatedFieldSentence(field)}` : field.description),
    ])),
    '',
    '**Concrete schemas are call-specific.** Run [`gpu_validate_call`](../../reference/gpu-function-catalog.mdx) against registered relations to inspect the output schema after the actual ID types and literal options are validated.',
  ].join('\n');
}

function quickstartPrerequisite(spec, entry) {
  const relations = Object.entries(entry.validation.call.relations || {});
  if (relations.length === 0) throw new Error(`${spec.function_name}: validation call names no relations`);
  const named = relations.map(([role, relation]) => `${code(relation.table)} (the ${code(role)} role)`);
  const passed = named.length > 1 ? 'each passed' : 'passed';
  return `The call below expects the registered relation${named.length > 1 ? 's' : ''} ${named.join(' and ')}, ${passed} as a parenthesized ${code('SELECT')} subquery. Substitute your own relations and column names.`;
}

function renderPage(spec) {
  const entry = presentation(spec);
  return renderFunctionPage({
    generator: 'website/scripts/gen-cuvs-docs.mjs',
    spec,
    title: entry.quickstart.display_name,
    quickstart: { prerequisite: quickstartPrerequisite(spec, entry), sql: entry.quickstart.sql },
    inputs: renderInputs(spec),
    argumentsAndOptions: renderArguments(spec),
    output: renderOutputSchema(spec),
    examples: null,
    limits: spec.limitations.map(limitation => `- ${prose(limitation)}`).join('\n'),
    discoveryLink: '../../reference/gpu-function-catalog.mdx',
  });
}

function renderCheatsheet(specs) {
  const sections = [];
  for (const [family, metadata] of Object.entries(FAMILIES)) {
    const familySpecs = specs.filter(spec => spec.family === family);
    if (familySpecs.length === 0) throw new Error(`presentation family ${family} has no function descriptor`);
    sections.push(
      `## ${metadata.label}`,
      '',
      table(['Function', 'Summary'], familySpecs.map(spec => {
        const entry = presentation(spec);
        return [`[${cell(entry.quickstart.display_name)}](./operations/${spec.function_name}.mdx)`, cell(spec.summary)];
      })),
    );
  }
  return [
    '{/* GENERATED FROM the unified GPU function catalog by website/scripts/gen-cuvs-docs.mjs. Do not edit by hand. */}',
    '',
    `The live catalog contains **${specs.length} SQL-visible cuVS functions**.`,
    '',
    ...sections,
  ].join('\n');
}

function main() {
  const specs = loadSpecs();
  const names = new Set(specs.map(spec => spec.function_name));
  for (const family of Object.values(FAMILIES)) {
    for (const functionName of family.functions) {
      if (!names.has(functionName)) throw new Error(`missing unified descriptor for ${functionName}`);
    }
  }
  for (const spec of specs) write(join(OPERATIONS_DIR, `${spec.function_name}.mdx`), renderPage(spec));
  write(join(DOCS_DIR, '_cheatsheet.mdx'), renderCheatsheet(specs));
  console.log(`Wrote ${specs.length} generated cuVS function pages and cheatsheet.`);
}

main();
