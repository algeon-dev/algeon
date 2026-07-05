const colors = require('tailwindcss/colors');

/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './docs/**/*.{md,mdx}',
    './src/**/*.{js,jsx,ts,tsx}',
    './node_modules/@tremor/react/dist/**/*.{js,cjs}',
  ],
  darkMode: ['class', '[data-theme="dark"]'],
  theme: {
    transparent: 'transparent',
    current: 'currentColor',
    extend: {
      colors: {
        'cpu-blue': '#0071C5',
        'gpu-green': '#76B900',
        tremor: {
          brand: {
            faint: colors.blue[50],
            muted: colors.blue[200],
            subtle: colors.blue[400],
            DEFAULT: colors.blue[500],
            emphasis: colors.blue[700],
            inverted: colors.white,
          },
          background: {
            muted: colors.gray[50],
            subtle: colors.gray[100],
            DEFAULT: colors.white,
            emphasis: colors.gray[700],
          },
          border: {
            DEFAULT: colors.gray[200],
          },
          ring: {
            DEFAULT: colors.gray[200],
          },
          content: {
            subtle: colors.gray[400],
            DEFAULT: colors.gray[500],
            emphasis: colors.gray[700],
            strong: colors.gray[900],
            inverted: colors.white,
          },
        },
        'dark-tremor': {
          brand: {
            faint: '#111827',
            muted: colors.amber[950],
            subtle: colors.amber[800],
            DEFAULT: colors.amber[500],
            emphasis: colors.amber[300],
            inverted: colors.gray[950],
          },
          background: {
            muted: '#0d1017',
            subtle: '#10141c',
            DEFAULT: '#141821',
            emphasis: colors.gray[300],
          },
          border: {
            DEFAULT: '#1b1f29',
          },
          ring: {
            DEFAULT: '#343b4a',
          },
          content: {
            subtle: '#5a6378',
            DEFAULT: '#8b95a3',
            emphasis: '#bfbdb6',
            strong: '#e6e1cf',
            inverted: colors.gray[950],
          },
        },
      },
      boxShadow: {
        'tremor-input': '0 1px 2px 0 rgb(0 0 0 / 0.05)',
        'tremor-card':
          '0 1px 3px 0 rgb(0 0 0 / 0.08), 0 1px 2px -1px rgb(0 0 0 / 0.08)',
        'tremor-dropdown':
          '0 4px 6px -1px rgb(0 0 0 / 0.08), 0 2px 4px -2px rgb(0 0 0 / 0.08)',
        'dark-tremor-input': '0 1px 2px 0 rgb(0 0 0 / 0.25)',
        'dark-tremor-card':
          '0 1px 3px 0 rgb(0 0 0 / 0.36), 0 1px 2px -1px rgb(0 0 0 / 0.36)',
        'dark-tremor-dropdown':
          '0 4px 6px -1px rgb(0 0 0 / 0.36), 0 2px 4px -2px rgb(0 0 0 / 0.36)',
      },
      borderRadius: {
        'tremor-small': '0.25rem',
        'tremor-default': '0.5rem',
        'tremor-full': '9999px',
      },
      fontSize: {
        'tremor-label': ['0.75rem', { lineHeight: '1rem' }],
        'tremor-default': ['0.875rem', { lineHeight: '1.25rem' }],
        'tremor-title': ['1.125rem', { lineHeight: '1.75rem' }],
        'tremor-metric': ['1.875rem', { lineHeight: '2.25rem' }],
      },
    },
  },
  safelist: [
    {
      pattern:
        /^(bg|text|border|ring|stroke|fill)-(slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-(50|100|200|300|400|500|600|700|800|900|950)$/,
      variants: ['dark', 'hover'],
    },
    {
      pattern: /^(bg|text|border|ring|stroke|fill)-(white|black|transparent)$/,
      variants: ['dark', 'hover'],
    },
    {
      pattern: /^(bg|text|border|ring|stroke|fill)-(cpu-blue|gpu-green)$/,
      variants: ['dark', 'hover'],
    },
  ],
};
