#!/usr/bin/env node
// Render captured cuML catalog descriptors without contacting a server.
// Refresh specs with website/scripts/capture-gpu-function-specs.sh first.
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  assertRenderableOptions,
  assertRenderableOutputFields,
  cell,
  code,
  densePresentation,
  formatFunctionQuickstartSql,
  loadProviderSpecs,
  narrativeExampleImport,
  narrativePartialImport,
  prose,
  renderDenseArguments,
  renderDenseOutputSchema,
  renderFunctionPage,
  renderVectorElementTypes,
  table,
  write,
} from './gpu-docs-common.mjs';

const WEBSITE = join(dirname(fileURLToPath(import.meta.url)), '..');
const SPEC_DIR = join(WEBSITE, 'gpu-function-specs', 'cuml');
const DOCS_DIR = join(WEBSITE, 'docs', 'cuml');
const UDTFS_DIR = join(DOCS_DIR, 'udtfs');
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
    'Each relation argument is a parenthesized `SELECT` subquery. Metadata validation resolves a registered table or view for the same role without scanning its rows. See [ML Inputs](../ml-inputs.mdx) for the ID, feature element, Float64 narrowing, null, finite-value, and runtime-dimension contract.',
    '',
    table(['Role', 'Required', 'Validation reference', 'Description'], spec.relation_roles.map(role => [
      code(role.name), role.required === false ? 'no' : 'yes', code(role.reference?.type || 'table'), cell(role.description),
    ])),
    '',
    '### Vector element types',
    '',
    renderVectorElementTypes(spec.relation_roles[0].fields.features.element_types.map(element => ({ element, metrics: [] }))),
  ].join('\n');
}

// Only the supervised UDTFs take a `training` relation with a label column;
// its type follows the family.
function renderBindingShapes(spec) {
  const label = spec.family === 'classification'
    ? ' The `training` relation also projects an Int32 `label`; the `predict` relation omits it.'
    : spec.family === 'regression'
      ? ' The `training` relation also projects a finite Float32 `label`; the `predict` relation omits it.'
      : '';
  const nulls = label ? 'A null ID, feature value, list row, list element, or label fails the query.' : 'A null ID, feature value, list row, or list element fails the query.';
  return [
    `Each relation subquery projects an \`id\` and either feature columns in dimension order or one list column, with the element types listed under Vector element types.${label} Nullable field types are accepted. ${nulls}`,
    '',
    'The list-column shape excludes other feature columns. See [ML Inputs](../ml-inputs.mdx) for the allowed list containers and runtime checks.',
  ].join('\n');
}

function renderPage(spec, sidebarPosition) {
  const entry = densePresentation(spec);
  const documentation = FUNCTIONS[spec.function_name];
  return renderFunctionPage({
    generator: 'website/scripts/gen-cuml-docs.mjs',
    spec,
    title: entry.quickstart.display_name,
    slug: `/cuml/udtfs/${spec.function_name}`,
    sidebarPosition,
    reference: `Official cuML reference: [${documentation.officialApi}](${officialRefUrl(documentation)})`,
    quickstart: {
      prerequisite: `Register the relations referenced by the parenthesized ${code('SELECT')} clauses below. For metadata validation, the descriptor names ${Object.values(entry.validation.call.relations).map(relation => code(relation.table)).join(' and ')}.`,
      sql: formatFunctionQuickstartSql(entry.quickstart.sql),
    },
    inputs: renderInputs(spec),
    argumentsAndOptions: renderDenseArguments(spec, renderBindingShapes(spec)),
    output: renderDenseOutputSchema(spec),
    notes: existsSync(join(DOCS_DIR, '_notes', `${spec.function_name}.mdx`))
      ? narrativePartialImport(`../_notes/${spec.function_name}.mdx`, 'Notes') : null,
    examples: existsSync(join(DOCS_DIR, '_examples', `${spec.function_name}.mdx`))
      ? narrativeExampleImport(`../_examples/${spec.function_name}.mdx`) : null,
    limits: spec.limitations.map(limitation => `- ${prose(limitation)}`).join('\n'),
  });
}

function renderCheatsheet(specs) {
  return [
    '{/* GENERATED FROM the unified GPU UDTF catalog by website/scripts/gen-cuml-docs.mjs. Do not edit by hand. */}',
    '',
    table(['UDTF', 'Family', 'Description', 'Official API'], specs.map(spec => {
      const documentation = FUNCTIONS[spec.function_name];
      return [
        `[${cell(densePresentation(spec).quickstart.display_name)}](./udtfs/${spec.function_name}.mdx)`,
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
  // The sidebar position is the cheatsheet row, so both list functions in the
  // same order.
  specs.forEach((spec, index) => {
    write(join(UDTFS_DIR, `${spec.function_name}.mdx`), renderPage(spec, index + 1));
  });
  write(join(DOCS_DIR, '_cheatsheet.mdx'), renderCheatsheet(specs));
  console.log(`Wrote ${specs.length} generated cuML UDTF pages and cheatsheet.`);
}

main();
