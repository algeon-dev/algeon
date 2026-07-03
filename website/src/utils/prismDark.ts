/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the Docusaurus source tree.
 */

import type {PrismTheme} from 'prism-react-renderer';

const prismDark = {
  plain: {
    color: '#bfbdb6',
    backgroundColor: '#10141c',
  },
  styles: [
    {
      types: ['comment', 'block-comment', 'prolog', 'doctype', 'cdata'],
      style: {
        color: '#7a8494',
        fontStyle: 'italic',
      },
    },
    {
      types: ['punctuation'],
      style: {
        color: '#f29668',
      },
    },
    {
      types: ['title'],
      style: {
        color: '#59c2ff',
        fontWeight: 'bold',
      },
    },
    {
      types: ['property', 'parameter', 'attr-name'],
      style: {
        color: '#59c2ff',
      },
    },
    {
      types: ['script'],
      style: {
        color: '#bfbdb6',
      },
    },
    {
      types: ['boolean', 'arrow', 'atrule', 'tag'],
      style: {
        color: '#39bae6',
      },
    },
    {
      types: ['number', 'color', 'unit'],
      style: {
        color: '#d2a6ff',
      },
    },
    {
      types: ['front-matter', 'front-matter-block'],
      style: {
        color: '#e6c08a',
      },
    },
    {
      types: ['keyword', 'rule'],
      style: {
        color: '#ff8f40',
      },
    },
    {
      types: ['regex', 'important'],
      style: {
        color: '#95e6cb',
      },
    },
    {
      types: ['maybe-class-name', 'class-name', 'builtin'],
      style: {
        color: '#aad94c',
      },
    },
    {
      types: ['constant'],
      style: {
        color: '#d2a6ff',
      },
    },
    {
      types: ['function'],
      style: {
        color: '#ffb454',
      },
    },
    {
      types: ['string', 'char'],
      style: {
        color: '#aad94c',
      },
    },
    {
      types: ['operator'],
      style: {
        color: '#f29668',
      },
    },
    {
      types: ['variable', 'symbol', 'selector'],
      style: {
        color: '#95e6cb',
      },
    },
    {
      types: ['deleted'],
      style: {
        color: '#f07178',
      },
    },
    {
      types: ['inserted'],
      style: {
        color: '#aad94c',
      },
    },
  ],
} satisfies PrismTheme;

export default prismDark;
