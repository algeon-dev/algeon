import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import prismDark from './src/utils/prismDark';
import prismLight from './src/utils/prismLight';
import highlightGpuCalls from './src/remark/highlightGpuCalls';
import {FOCUS_MODE_RESTORE_SCRIPT} from './src/components/focusMode/constants';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const GITHUB_ORG = 'algeon-dev';
const GITHUB_REPO = 'algeon';
const GITHUB_URL = `https://github.com/${GITHUB_ORG}/${GITHUB_REPO}`;

const config: Config = {
  title: 'Algeon',
  tagline: 'Compose GPU algorithms with SQL.',
  favicon: 'favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Production URL served from the apex custom domain through Cloudflare Pages.
  url: 'https://algeon.dev',
  // Served at the domain root (apex custom domain), so no repo-name prefix.
  baseUrl: '/',

  // Repository coordinates used by edit and source links.
  organizationName: GITHUB_ORG,
  projectName: GITHUB_REPO,

  onBrokenLinks: 'throw',

  clientModules: ['./src/clientModules/revealTabAnchor.ts'],

  headTags: [{tagName: 'script', attributes: {}, innerHTML: FOCUS_MODE_RESTORE_SCRIPT}],

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
          remarkPlugins: [highlightGpuCalls],
        },
        blog: {
          routeBasePath: 'blog',
          editUrl: `${GITHUB_URL}/tree/main/website/`,
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
      // Local, build-time, client-side search. No external service is involved.
      // The index is generated during `npm run build` and queried entirely in
      // the browser. The SearchBar appears in the navbar automatically.
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
    image: 'img/algeon-social-card.jpg',
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: false,
    },
    navbar: {
      // The wide logo already includes the "Algeon" wordmark, so the navbar
      // intentionally omits a separate title to avoid duplicating it.
      // The wordmark is a mid-tone slate that reads on both themes, so one
      // file serves light and dark without a `srcDark` variant.
      logo: {
        alt: 'Algeon',
        src: 'img/algeon-logo.png',
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
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'dark',
      copyright: 'Copyright © 2026 Algeon.',
    },
    prism: {
      theme: prismLight,
      darkTheme: prismDark,
      additionalLanguages: ['rust', 'toml', 'bash', 'cpp', 'c'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
