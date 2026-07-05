import {themes as prismThemes} from 'prism-react-renderer';
import type {PrismTheme} from 'prism-react-renderer';

const prismLight = {
  ...prismThemes.github,
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
