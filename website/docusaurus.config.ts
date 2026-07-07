import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import prismDark from './src/utils/prismDark';
import prismLight from './src/utils/prismLight';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const GITHUB_ORG = 'morristai';
const GITHUB_REPO = 'datafusion-nexus';
const GITHUB_URL = `https://github.com/${GITHUB_ORG}/${GITHUB_REPO}`;

const config: Config = {
  title: 'DataFusion Nexus',
  tagline: 'A DataFusion physical-plan adapter that lowers supported plans into a native, cuDF-backed query engine.',
  favicon: 'img/nexus-icon.png',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Production URL. Served from the apex custom domain `datafusion.nexus`
  // (Cloudflare-managed DNS) via GitHub Pages. The domain is pinned in
  // static/CNAME so GitHub Pages keeps serving it across deploys.
  url: 'https://datafusion.nexus',
  // Served at the domain root (apex custom domain), so no repo-name prefix.
  baseUrl: '/',

  // GitHub pages deployment config.
  organizationName: GITHUB_ORG,
  projectName: GITHUB_REPO,

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang.
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          // Serve content at the site root so the homepage is the Introduction.
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
        },
        blog: {
          routeBasePath: 'blog',
          editUrl: `${GITHUB_URL}/tree/main/website/blog/`,
          showReadingTime: false,
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: [
    [
      // Local, build-time, client-side search. No external service: the index
      // is generated during `npm run build` and queried entirely in the
      // browser. The SearchBar appears in the navbar automatically.
      //
      // To switch to Algolia DocSearch later, remove this `themes` block and
      // the `@easyops-cn/docusaurus-search-local` dependency, then add an
      // `algolia` entry under `themeConfig` (see README "Search").
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true, // append a content-hash query string to the index URL for cache busting
        indexDocs: true,
        // Match the route base path above.
        docsRouteBasePath: '/',
        indexBlog: true,
        indexPages: false, // content-first site; do not index standalone pages
        highlightSearchTermsOnTargetPage: true,
      },
    ],
  ],

  themeConfig: {
    // TODO: still the Docusaurus default template card. Replace with a
    // 1200x675 DataFusion Nexus branded social card before wide sharing.
    image: 'img/docusaurus-social-card.jpg',
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    navbar: {
      // The wide logo already includes the "DataFusion Nexus" wordmark, so we
      // intentionally omit a separate navbar title to avoid duplicating it.
      logo: {
        alt: 'DataFusion Nexus',
        src: 'img/nexus-logo-light-theme.png',
        srcDark: 'img/nexus-logo-dark-theme.png',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'right',
          label: 'Docs',
        },
        {
          to: '/blog',
          label: 'Blog',
          position: 'right',
        },
        {
          href: GITHUB_URL,
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      copyright: 'Copyright © 2026 DataFusion Nexus.',
    },
    prism: {
      theme: prismLight,
      darkTheme: prismDark,
      additionalLanguages: ['rust', 'toml', 'bash', 'cpp', 'c'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
