#!/usr/bin/env node
// Transform committed unified GPU catalog descriptors into the cuGraph SQL
// reference. The generator is deterministic and never contacts a server.
//
//   node website/scripts/gen-cugraph-docs.mjs
//
// Refresh descriptors with capture-cugraph-specs.sh before running this file.
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  cell,
  code,
  loadProviderSpecs,
  narrativeExampleImport,
  prose,
  renderFunctionPage,
  table,
  validationBlock,
  write,
} from './gpu-docs-common.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEBSITE = join(HERE, '..');
const SPEC_DIR = join(WEBSITE, 'gpu-function-specs', 'cugraph');
const DOCS_DIR = join(WEBSITE, 'docs', 'cugraph');
const ALGORITHMS_DIR = join(DOCS_DIR, 'algorithms');

// Presentation grouping remains specific to the cuGraph documentation; the
// function inventory itself always comes from the unified descriptors.
const FAMILIES = {
  centrality: { label: 'Centrality', position: 2, functions: ['cugraph_pagerank', 'cugraph_personalized_pagerank', 'cugraph_eigenvector_centrality', 'cugraph_katz_centrality', 'cugraph_hits', 'cugraph_betweenness_centrality', 'cugraph_edge_betweenness_centrality'] },
  community: { label: 'Community Detection', position: 3, functions: ['cugraph_louvain', 'cugraph_leiden', 'cugraph_ecg', 'cugraph_spectral_modularity_maximization'] },
  traversal: { label: 'Traversal & Paths', position: 4, functions: ['cugraph_bfs', 'cugraph_sssp'] },
  connectivity: { label: 'Connectivity', position: 5, functions: ['cugraph_weakly_connected_components', 'cugraph_strongly_connected_components', 'cugraph_minimum_spanning_tree'] },
  structure: { label: 'Structure', position: 6, functions: ['cugraph_degrees_all', 'cugraph_in_degrees_all', 'cugraph_out_degrees_all', 'cugraph_triangle_count_all', 'cugraph_core_number', 'cugraph_k_core'] },
  similarity: { label: 'Similarity', position: 7, functions: ['cugraph_jaccard', 'cugraph_sorensen', 'cugraph_overlap', 'cugraph_cosine'] },
  layout: { label: 'Layout', position: 1, functions: ['cugraph_force_atlas2'] },
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
    provider: 'cugraph',
    domain: 'graph',
    namePattern: /^cugraph_[a-z0-9_]+$/,
    validate(file, spec) {
      if (!Array.isArray(spec.relation_roles) || spec.relation_roles.length !== 1 || spec.relation_roles[0]?.name !== 'edges') {
        throw new Error(`${file}: cuGraph descriptor must declare exactly the edges relation role`);
      }
      if (!Array.isArray(spec.argument_descriptions) || !Array.isArray(spec.examples) || !Array.isArray(spec.limitations)) {
        throw new Error(`${file}: descriptor arrays have an invalid shape`);
      }
      if (!spec.options_schema || typeof spec.options_schema !== 'object' || !spec.result_schemas || typeof spec.result_schemas !== 'object') {
        throw new Error(`${file}: descriptor schemas must be JSON objects`);
      }
      spec.family = familyOf(spec.function_name);
    },
  });
}

function displayName(spec) {
  const words = new Map([
    ['bfs', 'BFS'], ['ecg', 'ECG'], ['hits', 'HITS'], ['sssp', 'SSSP'],
    ['pagerank', 'PageRank'], ['force_atlas2', 'ForceAtlas2'], ['k_core', 'K-Core'],
  ]);
  const name = spec.function_name.replace(/^cugraph_/, '');
  return words.get(name) ?? name
    .split('_')
    .map(word => words.get(word) ?? word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

const OPTION_DIRECT_KEYS = new Set(['type', 'default', 'description', 'scope']);
const OPTION_CONSTRAINT_KEYS = new Set([
  'allowed_values', 'min', 'max', 'exclusive_min', 'required', 'required_when',
  'required_with', 'mutually_exclusive_with', 'valid_when', 'defaults_to',
  'column_of', 'expected_type_ref', 'example', 'side_input',
]);

function optionConstraints(name, option) {
  for (const key of Object.keys(option)) {
    if (!OPTION_DIRECT_KEYS.has(key) && !OPTION_CONSTRAINT_KEYS.has(key)) {
      throw new Error(`option ${name} has unrendered descriptor key ${key}`);
    }
  }
  const parts = [];
  if (option.required === true) parts.push('required');
  if (option.allowed_values) parts.push(`one of ${option.allowed_values.map(value => code(JSON.stringify(value))).join(', ')}`);
  if ('min' in option) parts.push(`min ${option.min}`);
  if ('max' in option) parts.push(`max ${option.max}`);
  if ('exclusive_min' in option) parts.push(`> ${option.exclusive_min}`);
  if (option.required_when) parts.push(`required when ${option.required_when}`);
  if (option.required_with) parts.push(`required with ${code(option.required_with)}`);
  if (option.mutually_exclusive_with) parts.push(`mutually exclusive with ${code(option.mutually_exclusive_with)}`);
  if (option.valid_when) parts.push(`valid when ${option.valid_when}`);
  if (option.defaults_to) parts.push(`defaults to ${option.defaults_to}`);
  if (option.column_of) parts.push(`column of ${code(option.column_of)}`);
  if (option.expected_type_ref) parts.push(`type ref ${code(option.expected_type_ref)}`);
  if ('example' in option) parts.push(`example ${code(JSON.stringify(option.example))}`);
  if (option.side_input) {
    const columns = option.side_input.column_option ? [option.side_input.column_option] : (option.side_input.column_options || []);
    parts.push(`side input (${option.side_input.role}${columns.length ? `, cols: ${columns.map(code).join(', ')}` : ''})`);
  }
  return parts.map(cell).join('; ');
}

function optionTable(options) {
  return table(['Option', 'Type', 'Default', 'Constraints', 'Description'], Object.entries(options)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, option]) => [
      code(name),
      code(option.type),
      option.default === null ? code('null') : (Object.hasOwn(option, 'default') ? code(JSON.stringify(option.default)) : ''),
      optionConstraints(name, option),
      cell(option.description),
    ]));
}

function renderRelations(spec) {
  return [
    'cuGraph consumes a registered edge table or view by its first positional name. It does not accept a parenthesized relation subquery. Metadata validation uses the same registered name.',
    '',
    table(['Role', 'Required', 'Reference', 'Description'], spec.relation_roles.map(role => [
      code(role.name), role.required === false ? 'no' : 'yes', code(role.reference?.type || 'table'), cell(role.description),
    ])),
  ].join('\n');
}

function renderArguments(spec) {
  const argumentsTable = spec.argument_descriptions.length === 0
    ? '_This function has no non-relation arguments._'
    : table(['Argument', 'Type', 'Required', 'Default', 'Notes'], spec.argument_descriptions.map(argument => [
      code(argument.name),
      code(argument.type),
      argument.required ? 'yes' : 'no',
      Object.hasOwn(argument, 'default') ? code(argument.default) : '',
      cell([argument.note, argument.description, argument.semantic_effect && `semantic effect: ${argument.semantic_effect}`, argument.selector_policy && `selector: ${argument.selector_policy}`].filter(Boolean).join('; ')),
    ]));
  const algorithmOptions = spec.options_schema.algorithm_options || {};
  const optionSections = [
    '### Positional scalar arguments',
    '',
    argumentsTable,
    '',
    '### JSON options',
    '',
    Object.keys(algorithmOptions).length === 0 ? '_This function has no algorithm-specific options._' : optionTable(algorithmOptions),
  ];
  const graphOptions = spec.options_schema.graph_options || {};
  if (Object.keys(graphOptions).length > 0) {
    optionSections.push('', '### Graph construction options', '', 'Shared by all cuGraph functions, shown with this function\'s defaults. See [Graph Construction Options](../../graph-options.mdx) for the construction-policy guide.', '', optionTable(graphOptions));
  }
  return optionSections.join('\n');
}

function renderSchemas(spec) {
  const schemas = spec.result_schemas.schemas;
  if (!schemas || typeof schemas !== 'object') throw new Error(`${spec.function_name}: result_schemas_json must contain schemas`);
  const descriptions = Object.fromEntries((spec.result_schemas.field_descriptions || []).map(field => [field.name, field.description]));
  const defaultSchema = spec.result_schemas.default_schema;
  const names = Object.keys(schemas).sort((left, right) => (left === defaultSchema ? -1 : right === defaultSchema ? 1 : left.localeCompare(right)));
  const blocks = names.map(name => {
    const fields = schemas[name]?.fields;
    if (!Array.isArray(fields)) throw new Error(`${spec.function_name}: result schema ${name} has no fields`);
    const heading = names.length > 1 ? `### ${code(name)}${name === defaultSchema ? ' (default)' : ''}\n\n` : '';
    return heading + table(['Column', 'Type', 'Nullable', 'Description'], fields.map(field => [
      code(field.name), code(field.data_type), field.nullable ? 'yes' : 'no', cell(descriptions[field.name]),
    ]));
  });
  blocks.push('**Concrete schemas are call-specific.** These are generic descriptor schemas. Run [`gpu_validate_call`](../../../reference/gpu-function-catalog.mdx#gpu_validate_callfunction_name-call_json) for a concrete, table-specific output schema.');
  return blocks.join('\n\n');
}

function firstExample(spec) {
  const example = spec.examples.find(candidate => (candidate?.call || candidate?.call_json) && typeof (candidate.call || candidate.call_json) === 'object');
  if (!example) throw new Error(`${spec.function_name}: examples_json must contain a unified validation call`);
  if (typeof example.execution_sql !== 'string') throw new Error(`${spec.function_name}: example must contain execution_sql`);
  return { ...example, call: example.call || example.call_json };
}

function renderPage(spec) {
  const title = displayName(spec);
  const example = firstExample(spec);
  return renderFunctionPage({
    generator: 'website/scripts/gen-cugraph-docs.mjs',
    spec,
    title,
    useThisWhen: `Use this when a SQL statement needs ${title} results over a registered edge relation.`,
    quickstartSql: example.execution_sql,
    relationInputs: renderRelations(spec),
    scalarArgumentsAndOptions: renderArguments(spec),
    outputSchema: renderSchemas(spec),
    examples: narrativeExampleImport(`../../_examples/${spec.function_name}.mdx`),
    limitations: spec.limitations.map(limitation => `- ${prose(limitation)}`).join('\n'),
    validation: validationBlock({
      functionName: spec.function_name,
      call: example.call,
      discoveryLink: '../../../reference/gpu-function-catalog.mdx',
    }),
  });
}

function renderCheatsheet(specs) {
  return [
    '{/* GENERATED FROM the unified GPU function catalog by website/scripts/gen-cugraph-docs.mjs. Do not edit by hand. */}',
    '',
    `The live catalog contains **${specs.length} SQL-visible cuGraph functions**.`,
    '',
    table(['Function', 'Family', 'Signature', 'Summary'], specs.map(spec => [
      `[${cell(displayName(spec))}](./algorithms/${spec.family}/${spec.function_name}.mdx)`,
      cell(FAMILIES[spec.family].label),
      code(spec.function_name),
      cell(spec.summary),
    ])),
  ].join('\n');
}

function main() {
  const specs = loadSpecs();
  const classified = new Set(specs.map(spec => spec.function_name));
  for (const family of Object.values(FAMILIES)) {
    for (const functionName of family.functions) {
      if (!classified.has(functionName)) throw new Error(`missing unified descriptor for ${functionName}`);
    }
  }
  for (const spec of specs) {
    if (!Object.values(FAMILIES).some(family => family.functions.includes(spec.function_name))) {
      throw new Error(`unclassified unified descriptor ${spec.function_name}`);
    }
  }
  for (const [slug, family] of Object.entries(FAMILIES)) {
    write(join(ALGORITHMS_DIR, slug, '_category_.json'), JSON.stringify({ label: family.label, position: family.position }, null, 2));
  }
  for (const spec of specs) write(join(ALGORITHMS_DIR, spec.family, `${spec.function_name}.mdx`), renderPage(spec));
  write(join(DOCS_DIR, '_cheatsheet.mdx'), renderCheatsheet(specs));
  console.log(`Wrote ${specs.length} generated cuGraph function pages and cheatsheet.`);
}

main();
