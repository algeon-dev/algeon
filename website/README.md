# DataFusion Nexus Website

The documentation site for [DataFusion Nexus](https://github.com/morristai/datafusion-nexus),
built with [Docusaurus](https://docusaurus.io/) 3.10.1.

> Requires Node.js >= 20.

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
host.

```bash
npm run serve   # preview the production build locally
npm run typecheck
```

## Authoring docs

All documentation lives under `website/docs/`. Docs are served at the site root
(`docs.routeBasePath: '/'` in `docusaurus.config.ts`), so a file's URL mirrors
its path under `docs/` unless overridden by a `slug`.

**Adding a page.** Create a `.md` (plain Markdown) or `.mdx` (Markdown + React
components) file under `docs/`. Use `.mdx` when you need to import components,
e.g.:

```mdx
---
sidebar_position: 2
description: One-line summary used for SEO and category index cards.
---

import {StackDiagram} from '@site/src/components/design/Diagram';

# Page title
```

**Frontmatter.** Common fields used in this site:

- `sidebar_position` — orders the page within its sidebar group (lower first).
- `slug` — overrides the URL path (e.g. `slug: /` makes `intro.mdx` the homepage).
- `description` — short summary for SEO and generated category index pages.

**Sidebar.** The sidebar is auto-generated from the `docs/` tree
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

**Internal links.** Link to other docs by relative file path including the
extension (e.g. `[Architecture](./architecture/index.md)`); Docusaurus resolves
these at build time and `onBrokenLinks: 'throw'` fails the build on a bad link.

> **Note:** `website/` is tracked in git and deployed automatically (see
> [Deployment](#deployment)). Build artefacts (`build/`, `.docusaurus/`) and
> `node_modules/` are ignored via `website/.gitignore`.

## Search

Search is provided by
[`@easyops-cn/docusaurus-search-local`](https://github.com/easyops-cn/docusaurus-search-local),
a local, client-side search theme configured under `themes` in
`docusaurus.config.ts`. The search index is generated at build time and queried
entirely in the browser — there is no external search service, no account, and
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
migrate:

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
       indexName: 'datafusion-nexus',
       contextualSearch: true,
     },
   }
   ```

## Deployment

The site is deployed to **Cloudflare Pages** (project `datafusion-nexus`) at the
apex custom domain **<https://datafusion.nexus>** (`url` + `baseUrl: '/'` in
`docusaurus.config.ts`).

Two paths, both live:

- **Automatic** — `.github/workflows/deploy-website.yml` builds this directory
  and deploys via `cloudflare/wrangler-action` on every push to `main` that
  touches `website/**` (needs the `CLOUDFLARE_API_TOKEN` /
  `CLOUDFLARE_ACCOUNT_ID` repository secrets).
- **Manual** —
  `npm run build && npx wrangler pages deploy build --project-name=datafusion-nexus --branch=main`.

See [`DEPLOY.md`](./DEPLOY.md) for the full Cloudflare Pages setup, secrets, and
custom-domain reference.
