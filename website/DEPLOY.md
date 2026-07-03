# Deploying the website to GitHub Pages (custom domain `datafusion.nexus`)

The `website/` Docusaurus site is built and published to GitHub Pages by
`.github/workflows/deploy-website.yml` on every push to `main` that touches
`website/**`. It is served at the apex custom domain **https://datafusion.nexus**
(Cloudflare-managed DNS).

## One-time setup (dashboards)

Do these once, in order.

### 1. GitHub → Pages: enable Actions as the source

Repo → **Settings → Pages**:

- **Build and deployment → Source**: select **GitHub Actions**.

(Do NOT set the custom domain here yet — do it in step 3, after the first
successful deploy, so the `CNAME` file and the setting agree.)

### 2. Push, and let the workflow run

```bash
git push origin main
```

Watch **Actions → Deploy website**. The first run builds and publishes to the
`github-pages` environment. When green, the site is live at the default
`https://morristai.github.io/datafusion-nexus/`-style URL AND already carries the
`CNAME` file, so GitHub knows the intended custom domain.

### 3. Cloudflare DNS for zone `datafusion.nexus`

Cloudflare dashboard → zone `datafusion.nexus` → **DNS → Records**. Add:

| Type  | Name  | Target / Content        | Proxy status          |
|-------|-------|-------------------------|-----------------------|
| CNAME | `@`   | `morristai.github.io`   | **DNS only (grey)** ← |
| CNAME | `www` | `morristai.github.io`   | **DNS only (grey)**   |

Notes:

- Cloudflare **CNAME flattening** turns the apex `CNAME @` into the correct
  A/AAAA records automatically — you do NOT need to hardcode GitHub's four
  Pages IP addresses. (If you ever prefer explicit records instead, use A
  `185.199.108.153`, `.109.153`, `.110.153`, `.111.153` and their AAAA
  `2606:50c0:8000::153` … `8003::153`.)
- **Grey cloud (DNS only) is required initially.** GitHub Pages provisions a
  Let's Encrypt certificate for `datafusion.nexus` via an HTTP-01 challenge; if
  Cloudflare proxies (orange cloud) the record before that cert exists, the
  challenge and TLS fail. Keep it grey until GitHub shows the cert as issued.

### 4. GitHub → Pages: set + verify the custom domain

Back in **Settings → Pages**:

- **Custom domain**: enter `datafusion.nexus`, Save. GitHub runs a DNS check and
  begins issuing the TLS certificate (can take a few minutes to ~1 hour).
- Once the check passes and the cert is issued, tick **Enforce HTTPS**.

### 5. (Optional) Re-enable the Cloudflare proxy

After GitHub shows "certificate issued" and HTTPS is enforced, you MAY switch
the records to orange-cloud (proxied) for Cloudflare's CDN/caching/DDoS. If you
do, set Cloudflare **SSL/TLS mode = Full** (not Flexible) to avoid a redirect
loop. Leaving it grey-cloud is perfectly fine and simplest.

## Ongoing

Edit content under `website/docs/`, push to `main`, and the workflow redeploys.
The `CNAME` file (`website/static/CNAME`) is copied into every build, so the
custom domain persists across deploys.

## Local preview

```bash
cd website
npm ci
npm run build && npm run serve   # or: npm start  (dev server)
```
