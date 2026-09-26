#!/usr/bin/env node
// Render captured cuML catalog descriptors without contacting a server.
// Refresh specs with website/scripts/capture-gpu-function-specs.sh first.
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  assertRenderableOptions,
  assertRenderableOutputFields,
  cell,
  code,
  densePresentation,
  loadProviderSpecs,
  prose,
  renderDenseArguments,
  renderDenseOutputSchema,
  renderFunctionPage,
  table,
  write,
} from './gpu-docs-common.mjs';

const WEBSITE = join(dirname(fileURLToPath(import.meta.url)), '..');
const SPEC_DIR = join(WEBSITE, 'gpu-function-specs', 'cuml');
const DOCS_DIR = join(WEBSITE, 'docs', 'cuml');
const FUNCTIONS_DIR = join(DOCS_DIR, 'functions');
const OFFICIAL_DOCS_ROOT = 'https://docs.nvidia.com/cuml/latest/';

const FAMILIES = {
  dimensionality_reduction: { label: 'Dimensionality Reduction', functions: ['cuml_tsvd'] },
  clustering: { label: 'Clustering', functions: ['cuml_hdbscan'] },
  manifold: { label: 'Manifold Embedding', functions: ['cuml_umap'] },
  classification: { label: 'Classification', functions: ['cuml_logistic_regression', 'cuml_random_forest_classifier'] },
  regression: { label: 'Regression', functions: ['cuml_random_forest_regressor'] },
  inference: { label: 'Inference', functions: ['cuml_forest_predict'] },
};
const FUNCTIONS = {
  cuml_tsvd: {
    officialApi: 'Python API',
    officialPath: 'api/generated/cuml.decomposition.TruncatedSVD/',
  },
  cuml_hdbscan: {
    officialApi: 'Python API',
    officialPath: 'api/generated/cuml.cluster.hdbscan.HDBSCAN/',
  },
  cuml_umap: {
    officialApi: 'Python API',
    officialPath: 'api/generated/cuml.manifold.UMAP/',
  },
  cuml_logistic_regression: {
    officialApi: 'Python API',
    officialPath: 'api/generated/cuml.linear_model.LogisticRegression/',
  },
  cuml_random_forest_classifier: {
    officialApi: 'Python API',
    officialPath: 'api/generated/cuml.ensemble.RandomForestClassifier/',
  },
  cuml_random_forest_regressor: {
    officialApi: 'Python API',
    officialPath: 'api/generated/cuml.ensemble.RandomForestRegressor/',
  },
  cuml_forest_predict: {
    officialApi: 'nvForest',
    officialUrl: 'https://docs.nvidia.com/nvforest/latest/',
  },
};

function officialRefUrl(documentation) {
  return documentation.officialUrl || `${OFFICIAL_DOCS_ROOT}${documentation.officialPath}`;
}

function loadSpecs() {
  return loadProviderSpecs({
    specDir: SPEC_DIR,
    provider: 'cuml',
    namePattern: /^cuml_[a-z0-9_]+$/,
    validate(file, spec) {
      const family = Object.entries(FAMILIES).find(([, entry]) => entry.functions.includes(spec.function_name));
      if (!family || !Object.hasOwn(FUNCTIONS, spec.function_name)) {
        throw new Error(`${file}: unclassified cuML descriptor`);
      }
      spec.family = family[0];
      const wideSeed = spec.function_name === 'cuml_umap' ? 'random_state' :
        spec.function_name.startsWith('cuml_random_forest_') ? 'seed' : null;
      if (wideSeed) {
        // JSON.parse rounds u64::MAX; retain the captured schema's exact integer for documentation.
        const match = spec.options_schema_json.match(new RegExp(`"${wideSeed}":\\{[^}]*"maximum":([0-9]+)`));
        if (!match) throw new Error(`${file}: missing ${wideSeed} maximum`);
        spec.options_schema.properties[wideSeed].maximum = match[1];
      }
      if (!Array.isArray(spec.relation_roles) || spec.relation_roles.length === 0 ||
          spec.relation_roles.some(role => !role?.name || !role?.description ||
            !Array.isArray(role.fields?.features?.element_types) || role.fields.features.element_types.length === 0)) {
        throw new Error(`${file}: relation roles must describe dense feature elements`);
      }
      if (!Array.isArray(spec.argument_descriptions) || !Array.isArray(spec.examples) || !Array.isArray(spec.limitations) ||
          !spec.options_schema || typeof spec.options_schema !== 'object' ||
          !spec.result_schemas || !Array.isArray(spec.result_schemas.fields)) {
        throw new Error(`${file}: incomplete descriptor arrays or schemas`);
      }
      densePresentation(spec);
      assertRenderableOptions(spec);
      assertRenderableOutputFields(spec);
    },
  });
}

function renderInputs(spec) {
  return [
    'Each relation argument is a parenthesized `SELECT` subquery. Metadata validation resolves a registered table or view for the same role without scanning its rows. See [ML Inputs](../ml-inputs.mdx) for the ID, Float32 feature, null, finite-value, and runtime-dimension contract.',
    '',
    table(['Role', 'Required', 'Validation reference', 'Description'], spec.relation_roles.map(role => [
      code(role.name), role.required === false ? 'no' : 'yes', code(role.reference?.type || 'table'), cell(role.description),
    ])),
    '',
    '### Vector element types',
    '',
    table(['Element type', 'Valid metrics'], spec.relation_roles[0].fields.features.element_types.map(element => [
      code(element), 'Not applicable',
    ])),
  ].join('\n');
}

function renderPage(spec) {
  const entry = densePresentation(spec);
  const documentation = FUNCTIONS[spec.function_name];
  return renderFunctionPage({
    generator: 'website/scripts/gen-cuml-docs.mjs',
    spec,
    title: entry.quickstart.display_name,
    slug: `/cuml/functions/${spec.function_name}`,
    reference: `Official cuML reference: [${documentation.officialApi}](${officialRefUrl(documentation)})`,
    quickstart: {
      prerequisite: `Register the relations referenced by the parenthesized ${code('SELECT')} clauses below. For metadata validation, the descriptor names ${Object.values(entry.validation.call.relations).map(relation => code(relation.table)).join(' and ')}.`,
      sql: entry.quickstart.sql,
    },
    inputs: renderInputs(spec),
    argumentsAndOptions: renderDenseArguments(spec, [
      'Each relation subquery projects a non-null `id` and either non-null Float32 feature columns in dimension order or one non-null `vector` list column of non-null Float32 values. For classification, `training` also projects non-null Int32 `label`; regression requires non-null finite Float32 `label`. The `predict` relation omits it.',
      '',
      'The list-column shape excludes other feature columns. See [ML Inputs](../ml-inputs.mdx) for the allowed list containers and runtime checks.',
    ].join('\n')),
    output: renderDenseOutputSchema(spec),
    examples: null,
    limits: spec.limitations.map(limitation => `- ${prose(limitation)}`).join('\n'),
    discoveryLink: '../../reference/gpu-function-catalog.mdx',
  });
}

function renderCheatsheet(specs) {
  return [
    '{/* GENERATED FROM the unified GPU UDTF catalog by website/scripts/gen-cuml-docs.mjs. Do not edit by hand. */}',
    '',
    table(['Function', 'Family', 'Description', 'Official API'], specs.map(spec => {
      const documentation = FUNCTIONS[spec.function_name];
      return [
        `[${cell(densePresentation(spec).quickstart.display_name)}](./functions/${spec.function_name}.mdx)`,
        cell(FAMILIES[spec.family].label),
        cell(spec.summary),
        `[${cell(documentation.officialApi)}](${officialRefUrl(documentation)})`,
      ];
    })),
  ].join('\n');
}

function main() {
  const specs = loadSpecs();
  const names = new Set(specs.map(spec => spec.function_name));
  for (const family of Object.values(FAMILIES)) {
    for (const name of family.functions) {
      if (!names.has(name)) throw new Error(`missing unified descriptor for ${name}`);
    }
  }
  for (const spec of specs) write(join(FUNCTIONS_DIR, `${spec.function_name}.mdx`), renderPage(spec));
  write(join(DOCS_DIR, '_cheatsheet.mdx'), renderCheatsheet(specs));
  console.log(`Wrote ${specs.length} generated cuML function pages and cheatsheet.`);
}

main();
