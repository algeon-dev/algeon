# Algeon Website

The documentation site for [Algeon](https://github.com/algeon-dev/algeon).

`website/` is owned by the root repository. Source-build pages refer to the
recursive checkout, root Cargo workspace, and native artifacts under
`target/native/*`; component sources are pinned under `components/*`.

> Use the Node.js version declared in `package.json`.

## Install

```bash
npm install
```

## Local development

```bash
npm start
```

Starts a local dev server and opens a browser window. Most changes are
reflected live without restarting the server.

## Build

```bash
npm run build
```

Generates static content into the `build/` directory, servable by any static
host. The build first runs `npm run check:readme`, which compares the numbers
in the root `README.md` that carry a `<!-- capture:… -->` marker against the
demo captures in `static/data/`; a recapture that changes a result has to
update the README as well.

```bash
npm run serve   # preview the production build locally
npm run typecheck
```

## Authoring site content

The root [AGENTS.md](../AGENTS.md#contract-sources) routes engineering tasks to
the contract sections in `docs/`. Maintain those definitions in their owning
pages and link summaries to them. Contract changes update the documentation,
implementation and affected tests together. Machine-readable configuration and
owner-crate declarations supply exact values and API signatures; reconcile any
disagreement before changing behavior. Blog posts and illustrative examples
do not define implementation contracts.

Content pages are served at the site root, so a file's URL mirrors its source
tree path unless overridden by a `slug`.

**Adding a page.** Create a `.md` (plain Markdown) or `.mdx` (Markdown + React
components) file in the content tree. Use `.mdx` when you need to import
components, e.g.:

```mdx
---
sidebar_position: 2
description: One-line summary used for SEO and category index cards.
---

import {StackDiagram} from '@site/src/components/design/Diagram';

# Page title
```

**Frontmatter.** Common fields used in this site:

- `sidebar_position`: orders the page within its sidebar group (lower first).
- `slug`: overrides the URL path (e.g. `slug: /` makes `intro.mdx` the homepage).
- `description`: short summary for SEO and generated category index pages.

**Sidebar.** The sidebar is auto-generated from the content tree
(`sidebars.ts`). Folder grouping, label, and ordering are controlled by a
`_category_.json` in each subfolder, for example:

```json
{
  "label": "Architecture",
  "position": 2,
  "link": { "type": "doc", "id": "architecture/index" }
}
```

Use `"link": { "type": "generated-index", "description": "…" }` instead when the
category should land on an auto-generated index rather than a specific doc.

**Internal links.** Link to other pages by relative file path including the
extension (e.g. `[Design](./docs/design/index.mdx)`); Docusaurus resolves
these at build time and `onBrokenLinks: 'throw'` fails the build on a bad link.

> **Note:** `website/` is tracked in git and deployed automatically (see
> [Deployment](#deployment)). Build artefacts (`build/`, `.docusaurus/`) and
> `node_modules/` are ignored via `website/.gitignore`.

## Search

Search is provided by
[`@easyops-cn/docusaurus-search-local`](https://github.com/easyops-cn/docusaurus-search-local),
a local, client-side search theme configured under `themes` in
`docusaurus.config.ts`. The search index is generated at build time and queried
entirely in the browser; there is no external search service, no account, and
no public-crawlability requirement.

Two consequences to be aware of:

- **The index is built by `npm run build`, not `npm start`.** The dev server
  does not generate the search index, so search results are missing or stale in
  local development. To test search, run a production build and preview it:

  ```bash
  npm run build && npm run serve
  ```

- **Offline** refers to the search feature itself (no hosted backend). It is
  unrelated to the optional `open-ask-ai` "Ask AI" add-on, which *is* an
  external service and is **not installed** here.

### Switching to Algolia DocSearch later

Docusaurus officially integrates with [Algolia
DocSearch](https://docusaurus.io/docs/search#using-algolia-docsearch), which is
free for public/open-source technical docs but requires applying to the
DocSearch program and is only suitable once the site is publicly crawlable. To
switch:

1. Apply at <https://docsearch.algolia.com/apply> and wait for approval. You
   will receive an `appId`, a public `apiKey` (safe to commit), and an
   `indexName`; Algolia provisions and crawls the index for you.
2. Remove the `@easyops-cn/docusaurus-search-local` entry from the `themes`
   array in `docusaurus.config.ts`.
3. Remove the `@easyops-cn/docusaurus-search-local` dependency from
   `package.json` (then `npm install`).
4. Add an `algolia` entry under `themeConfig` (the Algolia search theme ships
   with `@docusaurus/preset-classic`, so no extra package is needed):

   ```ts
   themeConfig: {
     algolia: {
       appId: 'YOUR_APP_ID',
       apiKey: 'YOUR_SEARCH_ONLY_API_KEY',
       indexName: 'algeon',
       contextualSearch: true,
     },
   }
   ```

## Deployment

The site is deployed to **Cloudflare Pages** (project `algeon`) at the
apex custom domain **<https://algeon.dev>** (`url` + `baseUrl: '/'` in
`docusaurus.config.ts`).

Two paths, both live:

- **Automatic**: `.github/workflows/deploy-website.yml` builds this directory
  and deploys via `cloudflare/wrangler-action` on every push to `main` that
  touches `website/**` (needs the `CLOUDFLARE_API_TOKEN` /
  `CLOUDFLARE_ACCOUNT_ID` repository secrets).
- **Manual**:
  `npm ci && npm run typecheck && npm run build && npm run deploy`.

See [`DEPLOY.md`](./DEPLOY.md) for the full Cloudflare Pages setup, secrets, and
custom-domain reference.
