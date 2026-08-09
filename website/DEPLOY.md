# Deploying the website to Cloudflare Pages (`datafusion.nexus`)

The `website/` Docusaurus site in the DataFusion Nexus monorepo is deployed to **Cloudflare Pages** (project
`datafusion-nexus`) and served at the apex custom domain
**<https://datafusion.nexus>**. The domain lives in the same Cloudflare account
that hosts the Pages project, so DNS and TLS are managed entirely in Cloudflare.

There are two deploy paths, both live:

- **Manual** — `wrangler pages deploy` from your machine (fastest for testing).
- **Automatic** — `.github/workflows/deploy-website.yml` builds and deploys via
  `cloudflare/wrangler-action` on every push to `main` touching `website/**`
  (needs the two repository secrets below).

## Current state (already done)

- Pages project `datafusion-nexus` exists (`*.pages.dev`).
- Custom domain `datafusion.nexus` is attached and **active**, TLS issued by
  Google Trust Services, `http://` → `https://` redirect in place.
- Apex DNS: a **proxied (orange-cloud) `CNAME datafusion.nexus →
  datafusion-nexus.pages.dev`** in the `datafusion.nexus` zone. (Cloudflare
  Pages custom domains use the orange cloud — the opposite of GitHub Pages.)

## Manual deploy (from your machine)

```bash
cd website
npm ci                 # first time only
npm run build          # produces website/build
npx wrangler pages deploy build --project-name=datafusion-nexus --branch=main
```

`wrangler` uses your stored login (`npx wrangler login`) or a
`CLOUDFLARE_API_TOKEN` env var. `--branch=main` publishes to production; any
other branch name creates a preview deployment at a `*.pages.dev` alias.

## Automatic deploy (GitHub Actions)

The workflow builds Docusaurus then runs
`wrangler pages deploy build --project-name=datafusion-nexus`. It needs two
**repository secrets** (Settings → Secrets and variables → Actions):

| Secret | Value |
|--------|-------|
| `CLOUDFLARE_API_TOKEN` | A Cloudflare API token with **Account → Cloudflare Pages → Edit**. Create at My Profile → API Tokens. |
| `CLOUDFLARE_ACCOUNT_ID` | `YOUR_ACCOUNT_ID` |

Until both secrets exist, the deploy step fails (the build step still runs and
validates the site). Best set up once the repo has moved to its org, so the
token and secrets live where the repo does.

## Custom domain (already configured; for reference / re-creation)

If the domain ever needs re-attaching:

1. Cloudflare dashboard → **Workers & Pages → datafusion-nexus → Custom
   domains → Set up a custom domain** → `datafusion.nexus`. It offers to create
   the DNS record automatically.
2. Or manually, in the `datafusion.nexus` zone → **DNS → Records**: add
   `CNAME @ → datafusion-nexus.pages.dev`, **Proxied (orange cloud)**.

No `CNAME` file or `.nojekyll` handling is needed — those are GitHub Pages
concepts and do not apply to Cloudflare Pages.

## Local preview

```bash
cd website
npm ci
npm run build && npm run serve   # or: npm start  (dev server)
```
