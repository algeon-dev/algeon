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
const OFFICIAL_DOCS_ROOT = 'https://docs.rapids.ai/api/cugraph/stable/';

const FAMILIES = {
  centrality: { label: 'Centrality', position: 2 },
  community: { label: 'Community Detection', position: 3 },
  traversal: { label: 'Traversal & Paths', position: 4 },
  connectivity: { label: 'Connectivity', position: 5 },
  structure: { label: 'Structure', position: 6 },
  similarity: { label: 'Similarity', position: 7 },
  layout: { label: 'Layout', position: 1 },
};

const ALGORITHMS = {
  cugraph_betweenness_centrality: {
    family: 'centrality',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/centrality/#betweenness-centrality',
  },
  cugraph_bfs: {
    family: 'traversal',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/traversal/#breadth-first-search-bfs',
  },
  cugraph_core_number: {
    family: 'structure',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/core/#core-number',
  },
  cugraph_cosine: {
    family: 'similarity',
    officialApi: 'Python API',
    officialPath: 'api_docs/api/cugraph/cugraph.link_prediction.cosine/',
  },
  cugraph_degrees_all: {
    family: 'structure',
    officialApi: 'Python API',
    officialPath: 'api_docs/api/cugraph/cugraph.structure.graph_implementation.simpleGraphImpl.degrees/',
  },
  cugraph_ecg: {
    family: 'community',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/community/#ecg',
  },
  cugraph_edge_betweenness_centrality: {
    family: 'centrality',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/centrality/#edge-betweenness-centrality',
  },
  cugraph_eigenvector_centrality: {
    family: 'centrality',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/centrality/#eigenvector-centrality',
  },
  cugraph_force_atlas2: {
    family: 'layout',
    officialApi: 'Python API',
    officialPath: 'api_docs/api/cugraph/cugraph.force_atlas2/',
  },
  cugraph_hits: {
    family: 'centrality',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/centrality/#hits-centrality',
  },
  cugraph_in_degrees_all: {
    family: 'structure',
    officialApi: 'Python API',
    officialPath: 'api_docs/api/cugraph/cugraph.structure.graph_implementation.simpleGraphImpl.in_degree/',
  },
  cugraph_jaccard: {
    family: 'similarity',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/similarity/#jaccard',
  },
  cugraph_katz_centrality: {
    family: 'centrality',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/centrality/#katz-centrality',
  },
  cugraph_k_core: {
    family: 'structure',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/core/#k-core',
  },
  cugraph_leiden: {
    family: 'community',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/community/#leiden',
  },
  cugraph_louvain: {
    family: 'community',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/community/#louvain',
  },
  cugraph_minimum_spanning_tree: {
    family: 'connectivity',
    officialApi: 'Python API',
    officialPath: 'api_docs/api/cugraph/cugraph.tree.minimum_spanning_tree.minimum_spanning_tree/',
  },
  cugraph_out_degrees_all: {
    family: 'structure',
    officialApi: 'Python API',
    officialPath: 'api_docs/api/cugraph/cugraph.structure.graph_implementation.simpleGraphImpl.out_degree/',
  },
  cugraph_overlap: {
    family: 'similarity',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/similarity/#overlap',
  },
  cugraph_pagerank: {
    family: 'centrality',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/centrality/#pagerank',
  },
  cugraph_personalized_pagerank: {
    family: 'centrality',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/centrality/#personalized-pagerank',
  },
  cugraph_sorensen: {
    family: 'similarity',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/similarity/#sorensen',
  },
  cugraph_spectral_modularity_maximization: {
    family: 'community',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/community/#spectral-clustering-modularity-maximization',
  },
  cugraph_sssp: {
    family: 'traversal',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/traversal/#single-source-shortest-path-sssp',
  },
  cugraph_strongly_connected_components: {
    family: 'connectivity',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/labeling/#strongly-connected-components',
  },
  cugraph_triangle_count_all: {
    family: 'structure',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/community/#triangle-counting',
  },
  cugraph_weakly_connected_components: {
    family: 'connectivity',
    officialApi: 'C API',
    officialPath: 'api_docs/cugraph_c/labeling/#weakly-connected-components',
  },
};

const VERTEX_ID_CONTRACT_KEYS = [
  'numeric_endpoint_types',
  'logical_string_endpoint_types',
  'logical_string_output_type',
  'native_logical_mapping_type',
  'edge_id_supported_for_logical_string',
  'call_specific_authority',
];

function stringArray(file, value, key, { allowEmpty = true } = {}) {
  if (!Array.isArray(value) || (!allowEmpty && value.length === 0) || value.some(item => typeof item !== 'string' || item.length === 0)) {
    throw new Error(`${file}: vertex_id_contract.${key} must be an array of non-empty strings`);
  }
  return value;
}

function optionalStringArray(file, value, key) {
  if (value === undefined) return [];
  if (typeof value === 'string') {
    if (value.length === 0) throw new Error(`${file}: vertex_id_contract.${key} must contain non-empty strings`);
    return [value];
  }
  return stringArray(file, value, key);
}

function validateVertexIdContract(file, relationRole) {
  const contract = relationRole?.vertex_id_contract;
  if (!contract || typeof contract !== 'object' || Array.isArray(contract)) {
    throw new Error(`${file}: edges relation role is missing structured vertex_id_contract; recapture cuGraph descriptors before generating docs`);
  }
  for (const key of VERTEX_ID_CONTRACT_KEYS) {
    if (!(key in contract)) throw new Error(`${file}: vertex_id_contract is missing required key ${key}`);
  }
  stringArray(file, contract.numeric_endpoint_types, 'numeric_endpoint_types', { allowEmpty: false });
  const logicalEndpointTypes = stringArray(file, contract.logical_string_endpoint_types, 'logical_string_endpoint_types');
  const outputType = contract.logical_string_output_type;
  if (outputType !== null && typeof outputType !== 'string') {
    throw new Error(`${file}: vertex_id_contract.logical_string_output_type must be a string or null`);
  }
  const nativeMappingType = contract.native_logical_mapping_type;
  if (nativeMappingType !== null && typeof nativeMappingType !== 'string') {
    throw new Error(`${file}: vertex_id_contract.native_logical_mapping_type must be a string or null`);
  }
  if (logicalEndpointTypes.length > 0 && outputType === null) {
    throw new Error(`${file}: logical string endpoint support requires logical_string_output_type`);
  }
  if (logicalEndpointTypes.length === 0 && outputType !== null) {
    throw new Error(`${file}: logical_string_output_type requires logical string endpoint support`);
  }
  if (typeof contract.edge_id_supported_for_logical_string !== 'boolean') {
    throw new Error(`${file}: vertex_id_contract.edge_id_supported_for_logical_string must be boolean`);
  }
  if (typeof contract.call_specific_authority !== 'string' || contract.call_specific_authority.length === 0) {
    throw new Error(`${file}: vertex_id_contract.call_specific_authority must be a non-empty string`);
  }
  optionalStringArray(file, contract.logical_string_side_input_limitation, 'logical_string_side_input_limitation');
  optionalStringArray(file, contract.logical_string_side_input_limitations, 'logical_string_side_input_limitations');
  return contract;
}

function vertexIdContract(spec) {
  const role = spec.relation_roles.find(candidate => candidate.name === 'edges');
  if (!role) throw new Error(`${spec.function_name}: descriptor has no edges relation role`);
  return role.vertex_id_contract;
}

function documentationFor(functionName) {
  const documentation = ALGORITHMS[functionName];
  if (!documentation) throw new Error(`No documentation metadata for ${functionName}; classify it before generating docs`);
  return documentation;
}

function loadSpecs() {
  return loadProviderSpecs({
    specDir: SPEC_DIR,
    provider: 'cugraph',
    namePattern: /^cugraph_[a-z0-9_]+$/,
    validate(file, spec) {
      if (!Array.isArray(spec.relation_roles) || spec.relation_roles.length === 0 || spec.relation_roles[0]?.name !== 'edges') {
        throw new Error(`${file}: cuGraph descriptor must declare edges as its first relation role`);
      }
      const roleNames = new Set();
      for (const role of spec.relation_roles) {
        if (!role || typeof role.name !== 'string' || role.name.length === 0 || roleNames.has(role.name)) {
          throw new Error(`${file}: cuGraph descriptor relation roles must have unique non-empty names`);
        }
        if ((role.required !== undefined && typeof role.required !== 'boolean') || typeof role.description !== 'string') {
          throw new Error(`${file}: relation role ${role.name} has an invalid required or description field`);
        }
        roleNames.add(role.name);
      }
      validateVertexIdContract(file, spec.relation_roles[0]);
      if (!Array.isArray(spec.argument_descriptions) || !Array.isArray(spec.examples) || !Array.isArray(spec.limitations)) {
        throw new Error(`${file}: descriptor arrays have an invalid shape`);
      }
      if (!spec.options_schema || typeof spec.options_schema !== 'object' || !spec.result_schemas || typeof spec.result_schemas !== 'object') {
        throw new Error(`${file}: descriptor schemas must be JSON objects`);
      }
      spec.family = documentationFor(spec.function_name).family;
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

function renderInputs(spec) {
  const contract = vertexIdContract(spec);
  const logicalEndpointTypes = contract.logical_string_endpoint_types;
  const supportsLogicalStrings = logicalEndpointTypes.length > 0;
  const vertexIdLink = '[Vertex ID support](../../graph-options.mdx#vertex-id-support-by-function)';
  const parts = [];
  if (spec.relation_roles.length === 1) {
    parts.push(`${code('table_name')} must be a registered edge table or view (the ${code('edges')} role); parenthesized subqueries are not accepted, and metadata validation resolves the same registered name.`);
  } else {
    parts.push(
      `${code('table_name')} must be a registered edge table or view (the ${code('edges')} role); optional relations are named by their side-input table options. Parenthesized subqueries are not accepted, and metadata validation resolves the same registered names.`,
      '',
      table(['Role', 'Required', 'Contract'], spec.relation_roles.map(role => [
        code(role.name),
        role.required === false ? 'no' : 'yes',
        cell(role.description),
      ])),
    );
  }
  const dtypes = supportsLogicalStrings
    ? `Endpoint columns accept numeric ${renderTypeList(contract.numeric_endpoint_types)} vertex IDs or logical string ${renderTypeList(logicalEndpointTypes)} vertex IDs; string vertex-identity outputs are canonicalized to ${code(contract.logical_string_output_type)} (native mapping ${code(contract.native_logical_mapping_type)}) while scores, distances, counts, coordinates, and opaque labels stay numeric.`
    : `Endpoint columns accept ${renderTypeList(contract.numeric_endpoint_types)} vertex IDs only; logical string endpoints are not supported${contract.numeric_endpoint_types.length === 1 && contract.numeric_endpoint_types[0] === 'Int32' ? ' (legacy Int32-only contract)' : ''}.`;
  parts.push('', `${dtypes} The shared vertex-ID contract is summarized in ${vertexIdLink}; the concrete call-specific schema comes from ${code(contract.call_specific_authority)}.`);
  if (supportsLogicalStrings) {
    const limitations = logicalStringSideInputLimitations(spec, contract);
    parts.push('', 'Logical string side-input limitations:', '', limitations.length > 0 ? limitations.map(limitation => `- ${prose(limitation)}`).join('\n') : '- No additional logical string side-input limitation is declared.');
  }
  return parts.join('\n');
}

function renderTypeList(types) {
  return types.map(type => code(type)).join(', ');
}

function logicalStringSideInputLimitations(spec, contract) {
  const explicit = [
    ...optionalStringArray(spec.function_name, contract.logical_string_side_input_limitation, 'logical_string_side_input_limitation'),
  ];
  const plural = optionalStringArray(spec.function_name, contract.logical_string_side_input_limitations, 'logical_string_side_input_limitations');
  const limitations = [...explicit, ...plural];
  if (contract.edge_id_supported_for_logical_string === false && contract.logical_string_endpoint_types.length > 0) {
    limitations.push('edge ID columns and edge-ID predicate side inputs are not supported for logical string graphs');
  }
  const proseLimitations = spec.limitations.filter(limitation =>
    /string/i.test(limitation) && /(side|input|edge[_ -]?id|predicate|source|seed|candidate|personalization)/i.test(limitation));
  limitations.push(...proseLimitations);
  return [...new Set(limitations)];
}


function renderLimitations(spec) {
  const limitations = spec.limitations.filter(limitation => !limitation.startsWith('dry-run'));
  if (limitations.length === 0) return '_No algorithm-specific limitations._';
  return limitations.map(limitation => `- ${prose(limitation)}`).join('\n');
}

function isSharedEndpointArgument(argument) {
  const canonical = (argument.name === 'src_col' && argument.default === 'src')
    || (argument.name === 'dst_col' && argument.default === 'dst');
  return canonical
    && argument.type === 'Utf8'
    && argument.required === false
    && !argument.note && !argument.description && !argument.semantic_effect && !argument.selector_policy;
}

function renderGraphOptionsNote(spec) {
  const directed = spec.options_schema.graph_options?.directed;
  const link = '[Graph Construction Options](../../graph-options.mdx)';
  if (directed?.allowed_values?.length === 1 && directed.allowed_values[0] === false) {
    return `This function requires ${code('directed=false')} (undirected/symmetric graph); all other graph construction options follow the shared defaults documented in ${link}.`;
  }
  if (directed?.default === false) {
    return `This function builds an undirected graph by default (${code('directed=false')}); all other graph construction options follow the shared defaults documented in ${link}.`;
  }
  return `Graph construction follows the shared defaults (${code('directed=true')}, renumbering, ${code('python_cugraph')} policy) documented in ${link}.`;
}

function renderArguments(spec) {
  const sharedEndpoints = spec.argument_descriptions.filter(isSharedEndpointArgument);
  const useEndpointSentence = sharedEndpoints.length === 2;
  const rows = useEndpointSentence ? spec.argument_descriptions.filter(argument => !isSharedEndpointArgument(argument)) : spec.argument_descriptions;
  const positionalSections = [];
  if (useEndpointSentence) {
    positionalSections.push(`${code('src_col')} and ${code('dst_col')} name the edge endpoint columns; both are optional and default to ${code('src')} and ${code('dst')}.`);
  }
  if (rows.length > 0) {
    positionalSections.push(table(['Argument', 'Type', 'Required', 'Default', 'Notes'], rows.map(argument => [
      code(argument.name),
      code(argument.type),
      argument.required ? 'yes' : 'no',
      Object.hasOwn(argument, 'default') ? code(argument.default) : '',
      cell([argument.note, argument.description, argument.semantic_effect && `semantic effect: ${argument.semantic_effect}`, argument.selector_policy && `selector: ${argument.selector_policy}`].filter(Boolean).join('; ')),
    ])));
  } else if (!useEndpointSentence) {
    positionalSections.push('_This function has no non-relation arguments._');
  }
  const algorithmOptions = spec.options_schema.algorithm_options || {};
  const optionSections = [
    '### Positional scalar arguments',
    '',
    positionalSections.join('\n\n'),
    '',
    '### JSON options',
    '',
    Object.keys(algorithmOptions).length === 0 ? '_This function has no algorithm-specific options._' : optionTable(algorithmOptions),
  ];
  const graphOptions = spec.options_schema.graph_options || {};
  if (Object.keys(graphOptions).length > 0) {
    optionSections.push('', '### Graph construction options', '', renderGraphOptionsNote(spec));
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
  blocks.push('These are generic descriptor schemas; [validate the call](#validate-the-call) to get the concrete, table-specific output schema.');
  return blocks.join('\n\n');
}

function firstExample(spec) {
  const example = spec.examples.find(candidate => (candidate?.call || candidate?.call_json) && typeof (candidate.call || candidate.call_json) === 'object');
  if (!example) throw new Error(`${spec.function_name}: examples_json must contain a unified validation call`);
  return { ...example, call: example.call || example.call_json };
}

function quickstartPrerequisite(spec, example) {
  const options = example.call.options || {};
  const edges = example.call.relations?.edges?.table;
  if (typeof edges !== 'string' || edges.length === 0) throw new Error(`${spec.function_name}: example call names no edges table`);
  let sentence = `The call below expects a registered edge table or view ${code(edges)} with endpoint columns ${code(options.src_col ?? 'src')} and ${code(options.dst_col ?? 'dst')}`;
  if (typeof options.weight_col === 'string') sentence += ` and weight column ${code(options.weight_col)}`;
  const sideInputs = [];
  for (const [name, option] of Object.entries(spec.options_schema.algorithm_options || {})) {
    if (!option.side_input || typeof options[name] !== 'string') continue;
    const columnOptions = option.side_input.column_option ? [option.side_input.column_option] : (option.side_input.column_options || []);
    const columns = columnOptions.map(columnOption => options[columnOption]).filter(value => typeof value === 'string');
    sideInputs.push(`${code(options[name])} (the ${code(option.side_input.role)} side input${columns.length ? ` with columns ${columns.map(code).join(', ')}` : ''})`);
  }
  if (sideInputs.length > 0) sentence += `, plus the registered relation${sideInputs.length > 1 ? 's' : ''} ${sideInputs.join(' and ')}`;
  if (options.source_vertex !== undefined && options.source_vertex !== null) {
    sentence += `, and starts from source vertex ${code(JSON.stringify(options.source_vertex))}`;
  }
  return `${sentence}. Substitute your own registered relations.`;
}

function renderPage(spec) {
  const title = displayName(spec);
  const example = firstExample(spec);
  if (typeof example.execution_sql !== 'string' || example.execution_sql.trim() === '') {
    throw new Error(`${spec.function_name}: examples_json must contain execution_sql for the quickstart`);
  }
  const documentation = documentationFor(spec.function_name);
  const officialDocsUrl = `${OFFICIAL_DOCS_ROOT}${documentation.officialPath}`;
  return renderFunctionPage({
    generator: 'website/scripts/gen-cugraph-docs.mjs',
    spec,
    title,
    reference: `Official cuGraph reference: [${documentation.officialApi}](${officialDocsUrl})`,
    quickstart: { prerequisite: quickstartPrerequisite(spec, example), sql: `${example.execution_sql.trim().replace(/;$/, '')};` },
    inputs: renderInputs(spec),
    argumentsAndOptions: renderArguments(spec),
    output: renderSchemas(spec),
    examples: narrativeExampleImport(`../../_examples/${spec.function_name}.mdx`),
    limits: renderLimitations(spec),
    validation: validationBlock({
      functionName: spec.function_name,
      call: example.call,
      discoveryLink: '../../../reference/gpu-function-catalog.mdx',
      intro: 'Dry-run validation checks registered relation metadata, column presence, static dtypes, and options only; it does not scan edge data, construct a graph, or prove source-vertex existence:',
    }),
  });
}

function renderCheatsheet(specs) {
  return [
    '{/* GENERATED FROM the unified GPU function catalog by website/scripts/gen-cugraph-docs.mjs. Do not edit by hand. */}',
    '',
    `The live catalog contains **${specs.length} SQL-visible cuGraph functions**.`,
    '',
    table(['Function', 'Family', 'Signature', 'Summary', 'Official API'], specs.map(spec => {
      const documentation = documentationFor(spec.function_name);
      return [
        `[${cell(displayName(spec))}](./algorithms/${spec.family}/${spec.function_name}.mdx)`,
        cell(FAMILIES[spec.family].label),
        code(spec.function_name),
        cell(spec.summary),
        `[${cell(documentation.officialApi)}](${OFFICIAL_DOCS_ROOT}${documentation.officialPath})`,
      ];
    })),
  ].join('\n');
}

function renderVertexIdSupport(specs) {
  const rows = specs.map(spec => {
    const contract = vertexIdContract(spec);
    const logicalEndpointTypes = contract.logical_string_endpoint_types;
    const supportsLogicalStrings = logicalEndpointTypes.length > 0;
    const sideInputLimitations = logicalStringSideInputLimitations(spec, contract);
    const legacyInt32Only = contract.numeric_endpoint_types.length === 1
      && contract.numeric_endpoint_types[0] === 'Int32'
      && !supportsLogicalStrings;
    return [
      `[${cell(displayName(spec))}](./algorithms/${spec.family}/${spec.function_name}.mdx)`,
      renderTypeList(contract.numeric_endpoint_types),
      supportsLogicalStrings ? renderTypeList(logicalEndpointTypes) : 'Not supported',
      supportsLogicalStrings ? code(contract.logical_string_output_type) : 'None',
      supportsLogicalStrings ? sideInputLimitations.map(prose).join('; ') || 'None declared' : 'Not applicable',
      legacyInt32Only ? 'Yes' : 'No',
    ];
  });
  return [
    '{/* GENERATED FROM the unified GPU function catalog by website/scripts/gen-cugraph-docs.mjs. Do not edit by hand. */}',
    '',
    '## Vertex ID support by function',
    '',
    'This matrix is generated from each function descriptor’s structured `edges.vertex_id_contract`. It is the authoritative overview of numeric endpoint inputs, logical string endpoint inputs, canonical string output, side-input limitations, and legacy Int32-only dispatch.',
    '',
    'Logical string input means any `Utf8`, `LargeUtf8`, or `Utf8View` edge endpoint. The physical string types may be mixed within one graph call; vertex-identity outputs use the declared canonical string type. Numeric endpoint requirements remain function-specific and are not widened by logical string support.',
    '',
    table(['Function', 'Numeric endpoint types', 'Logical string endpoint types', 'Canonical string output', 'Logical side-input limitation', 'Legacy Int32-only'], rows),
    '',
    'Use `gpu_describe_function` for the full descriptor and `gpu_validate_call` for the concrete call-specific schema and side-input validation.',
  ].join('\n');
}

function main() {
  const specs = loadSpecs();
  const classified = new Set(specs.map(spec => spec.function_name));
  for (const functionName of Object.keys(ALGORITHMS)) {
    if (!classified.has(functionName)) throw new Error(`missing unified descriptor for ${functionName}`);
  }
  for (const spec of specs) {
    if (!Object.hasOwn(ALGORITHMS, spec.function_name)) {
      throw new Error(`unclassified unified descriptor ${spec.function_name}`);
    }
  }
  for (const [slug, family] of Object.entries(FAMILIES)) {
    write(join(ALGORITHMS_DIR, slug, '_category_.json'), JSON.stringify({ label: family.label, position: family.position }, null, 2));
  }
  for (const spec of specs) write(join(ALGORITHMS_DIR, spec.family, `${spec.function_name}.mdx`), renderPage(spec));
  write(join(DOCS_DIR, '_cheatsheet.mdx'), renderCheatsheet(specs));
  write(join(DOCS_DIR, '_vertex-id-support.mdx'), renderVertexIdSupport(specs));
  console.log(`Wrote ${specs.length} generated cuGraph function pages, cheatsheet, and vertex-ID support partial.`);
}

main();
