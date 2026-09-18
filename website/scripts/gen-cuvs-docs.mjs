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
  assertRenderableOptions,
  assertRenderableOutputFields,
  densePresentation as presentation,
  denseQuickstartPrerequisite as quickstartPrerequisite,
  loadProviderSpecs,
  renderDenseArguments as renderArguments,
  renderDenseOutputSchema as renderOutputSchema,
  prose,
  renderFunctionPage,
  table,
  write,
} from './gpu-docs-common.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const WEBSITE = join(HERE, '..');
const SPEC_DIR = join(WEBSITE, 'gpu-function-specs', 'cuvs');
const DOCS_DIR = join(WEBSITE, 'docs', 'cuvs');
const FUNCTIONS_DIR = join(DOCS_DIR, 'functions');
const OFFICIAL_DOCS_ROOT = 'https://docs.nvidia.com/cuvs/';

// These presentation families are documentation navigation only. They do not
// alter SQL names or introduce an execution API family.
const FAMILIES = {
  nearest_neighbors: {
    label: 'Nearest Neighbors',
    functions: ['cuvs_brute_force_knn', 'cuvs_ivf_flat', 'cuvs_ivf_pq', 'cuvs_cagra'],
  },
  distance: {
    label: 'Distance',
    functions: ['cuvs_pairwise_distance'],
  },
  selection: {
    label: 'Selection',
    functions: ['cuvs_top_k'],
  },
  cluster: {
    label: 'Cluster',
    functions: ['cuvs_kmeans'],
  },
  preprocessing: {
    label: 'Preprocessing',
    functions: ['cuvs_pca'],
  },
};

const FUNCTIONS = {
  cuvs_brute_force_knn: {
    officialApi: 'C API',
    officialPath: 'api-reference/c-api-neighbors-brute-force',
  },
  cuvs_cagra: {
    officialApi: 'C API',
    officialPath: 'api-reference/c-api-neighbors-cagra',
  },
  cuvs_ivf_flat: {
    officialApi: 'C API',
    officialPath: 'api-reference/c-api-neighbors-ivf-flat',
  },
  cuvs_ivf_pq: {
    officialApi: 'C API',
    officialPath: 'api-reference/c-api-neighbors-ivf-pq',
  },
  cuvs_pairwise_distance: {
    officialApi: 'C API',
    officialPath: 'api-reference/c-api-distance-pairwise-distance',
  },
  cuvs_top_k: {
    officialApi: 'C++ API',
    officialPath: 'api-reference/cpp-api-selection-select-k',
  },
  cuvs_kmeans: {
    officialApi: 'C API',
    officialPath: 'api-reference/c-api-cluster-kmeans',
  },
  cuvs_pca: {
    officialApi: 'C API',
    officialPath: 'api-reference/c-api-preprocessing-pca',
  },
};

function documentationFor(functionName) {
  const documentation = FUNCTIONS[functionName];
  if (!documentation) {
    throw new Error(`missing official documentation metadata for ${functionName}`);
  }
  return documentation;
}

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
      const support = spec.relation_roles[0]?.fields?.features?.element_metric_support;
      if (!Array.isArray(support) || JSON.stringify(support.map(entry => entry.element)) !== JSON.stringify(elements) ||
          support.some(entry => !Array.isArray(entry.metrics)) ||
          spec.relation_roles.some(role => JSON.stringify(role.fields?.features?.element_metric_support) !== JSON.stringify(support))) {
        throw new Error(`${file}: vector element/metric support must match across relation roles`);
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
    table(['Element type', 'Valid metrics'], spec.relation_roles[0].fields.features.element_metric_support.map(support => [
      code(support.element), support.metrics.length ? support.metrics.map(code).join(', ') : 'Not applicable',
    ])),
  ].join('\n');
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

function renderPage(spec) {
  const entry = presentation(spec);
  const documentation = documentationFor(spec.function_name);
  const officialDocsUrl = `${OFFICIAL_DOCS_ROOT}${documentation.officialPath}`;
  return renderFunctionPage({
    generator: 'website/scripts/gen-cuvs-docs.mjs',
    spec,
    title: entry.quickstart.display_name,
    slug: `/cuvs/functions/${spec.function_name}`,
    reference: `Official cuVS reference: [${documentation.officialApi}](${officialDocsUrl})`,
    quickstart: { prerequisite: quickstartPrerequisite(spec, entry), sql: entry.quickstart.sql },
    inputs: renderInputs(spec),
    argumentsAndOptions: renderArguments(spec, renderVectorBindingShapes(spec)),
    output: renderOutputSchema(spec),
    examples: null,
    limits: spec.limitations.map(limitation => `- ${prose(limitation)}`).join('\n'),
    discoveryLink: '../../reference/gpu-function-catalog.mdx',
  });
}

function renderCheatsheet(specs) {
  return [
    '{/* GENERATED FROM the unified GPU UDTF catalog by website/scripts/gen-cuvs-docs.mjs. Do not edit by hand. */}',
    '',
    table(['Function', 'Family', 'Description', 'Official API'], specs.map(spec => {
      const entry = presentation(spec);
      const documentation = documentationFor(spec.function_name);
      return [
        `[${cell(entry.quickstart.display_name)}](./functions/${spec.function_name}.mdx)`,
        cell(FAMILIES[spec.family].label),
        cell(spec.summary),
        `[${cell(documentation.officialApi)}](${OFFICIAL_DOCS_ROOT}${documentation.officialPath})`,
      ];
    })),
  ].join('\n');
}

function renderDtypeSupport(specs) {
  const elements = ['Float32', 'Float64', 'Int8', 'UInt8'];
  return [
    '{/* GENERATED FROM the unified GPU UDTF catalog by website/scripts/gen-cuvs-docs.mjs. Do not edit by hand. */}',
    '',
    '## Vector element and metric support by function',
    '',
    'Each cell lists valid `metric` values for that input element. Yes means the element is supported without a `metric` option; No means the element is unsupported.',
    '',
    table(['Function', ...elements], specs.map(spec => {
      const support = spec.relation_roles[0].fields.features.element_metric_support;
      return [code(spec.function_name), ...elements.map(element => {
        const entry = support.find(candidate => candidate.element === element);
        return !entry ? 'No' : entry.metrics.length ? entry.metrics.map(code).join(', ') : 'Yes';
      })];
    })),
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
  for (const functionName of Object.keys(FUNCTIONS)) {
    if (!names.has(functionName)) throw new Error(`missing unified descriptor for ${functionName}`);
  }
  for (const spec of specs) {
    if (!Object.hasOwn(FUNCTIONS, spec.function_name)) {
      throw new Error(`unclassified unified descriptor ${spec.function_name}`);
    }
  }
  for (const spec of specs) write(join(FUNCTIONS_DIR, `${spec.function_name}.mdx`), renderPage(spec));
  write(join(DOCS_DIR, '_cheatsheet.mdx'), renderCheatsheet(specs));
  write(join(DOCS_DIR, '_vector-element-support.mdx'), renderDtypeSupport(specs));
  console.log(`Wrote ${specs.length} generated cuVS function pages and cheatsheet.`);
}

main();
