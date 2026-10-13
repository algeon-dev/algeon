import {themes as prismThemes} from 'prism-react-renderer';
import type {PrismTheme} from 'prism-react-renderer';

const prismLight = {
  ...prismThemes.github,
  plain: {
    ...prismThemes.github.plain,
    // GitHub's cool gray would sit on the warm paper page; custom.css owns
    // the light code surface.
    backgroundColor: 'var(--ifm-code-background)',
  },
  styles: [
    ...prismThemes.github.styles,
    {
      types: ['key', 'property'],
      languages: ['toml'],
      style: {
        color: '#393A34',
      },
    },
    {
      types: ['variable'],
      languages: ['bash', 'sh', 'shell'],
      style: {
        color: '#1f6feb',
      },
    },
  ],
} satisfies PrismTheme;

export default prismLight;
