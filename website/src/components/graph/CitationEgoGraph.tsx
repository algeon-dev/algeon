import React, {useEffect, useRef, useState} from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import {useColorMode} from '@docusaurus/theme-common';
import data from './attentionEgo';
import styles from './styles.module.css';

// Categorical slots (validated light/dark pairs); the 6th group folds into gray.
const GROUP_COLORS: Record<string, {light: string; dark: string}> = {
  'Transformer core & lineage': {light: '#2a78d6', dark: '#59c2ff'},
  'Neural machine translation': {light: '#1baf7a', dark: '#95e6cb'},
  'BERT & pretraining': {light: '#eda100', dark: '#ffb454'},
  'Vision & scaled training': {light: '#008300', dark: '#aad94c'},
  'Self-attention applications': {light: '#4a3aa7', dark: '#d2a6ff'},
  Other: {light: '#8a8880', dark: '#8b95a3'},
};

function color(group: string, mode: 'light' | 'dark'): string {
  return (GROUP_COLORS[group] ?? GROUP_COLORS.Other)[mode];
}

type HoverNode = {
  x: number;
  y: number;
  size: number;
  label: string | null;
  color: string;
};

type HoverSettings = {
  labelFont: string;
  labelSize: number;
  labelWeight: string;
};

function drawNodeHover(
  context: CanvasRenderingContext2D,
  node: HoverNode,
  settings: HoverSettings,
  dark: boolean,
): void {
  const size = settings.labelSize;
  const padding = 2;
  const background = dark ? '#141821' : '#ffffff';
  const shadow = dark ? 'rgba(0, 0, 0, 0.55)' : '#000000';
  const text = dark ? '#e6e1cf' : '#0b0b0b';

  context.font = `${settings.labelWeight} ${size}px ${settings.labelFont}`;
  context.fillStyle = background;
  context.shadowOffsetX = 0;
  context.shadowOffsetY = 0;
  context.shadowBlur = 8;
  context.shadowColor = shadow;

  if (typeof node.label === 'string') {
    const textWidth = context.measureText(node.label).width;
    const boxWidth = Math.round(textWidth + 5);
    const boxHeight = Math.round(size + 2 * padding);
    const radius = Math.max(node.size, size / 2) + padding;
    const angleRadian = Math.asin(boxHeight / 2 / radius);
    const xDeltaCoord = Math.sqrt(Math.abs(radius ** 2 - (boxHeight / 2) ** 2));

    context.beginPath();
    context.moveTo(node.x + xDeltaCoord, node.y + boxHeight / 2);
    context.lineTo(node.x + radius + boxWidth, node.y + boxHeight / 2);
    context.lineTo(node.x + radius + boxWidth, node.y - boxHeight / 2);
    context.lineTo(node.x + xDeltaCoord, node.y - boxHeight / 2);
    context.arc(node.x, node.y, radius, angleRadian, -angleRadian);
    context.closePath();
    context.fill();
  } else {
    context.beginPath();
    context.arc(node.x, node.y, node.size + padding, 0, Math.PI * 2);
    context.closePath();
    context.fill();
  }

  context.shadowBlur = 0;
  context.fillStyle = text;
  context.fillText(node.label ?? '', node.x + node.size + 3, node.y + size / 3);
}

function EgoCanvas(): React.ReactElement {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const {colorMode} = useColorMode();
  const [hovered, setHovered] = useState<string | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (container == null) {
      return undefined;
    }
    let disposed = false;
    let renderer: {kill: () => void} | null = null;

    Promise.all([import('graphology'), import('sigma')]).then(
      ([{default: Graph}, {default: Sigma}]) => {
        if (disposed || containerRef.current == null) {
          return;
        }
        const maxDegree = Math.max(...data.nodes.map((n) => n.degree));
        const graph = new Graph();
        for (const n of data.nodes) {
          graph.addNode(n.id, {
            x: n.x,
            y: n.y,
            size: 3 + 11 * Math.sqrt(n.degree / maxDegree),
            label: `${n.title} (${n.year})`,
            color: color(n.group, colorMode),
          });
        }
        for (const [src, dst] of data.edges) {
          if (!graph.hasEdge(src, dst)) {
            graph.addEdge(src, dst);
          }
        }
        const dark = colorMode === 'dark';
        // Solid hex only: sigma's WebGL color packing does not take rgba() strings.
        const edgeColor = dark ? '#343b4a' : '#d9d8d3';
        const dimmed = dark ? '#1b1f29' : '#e4e3de';
        const highlight = dark ? '#e6b450' : '#52514e';
        // Mutable hover state closed over by the reducers: the reducers run
        // during Sigma's constructor render, before the instance binding exists.
        const state: {hovered?: string} = {};
        const sigma = new Sigma(graph, container, {
          renderLabels: false,
          defaultEdgeColor: edgeColor,
          labelColor: {color: dark ? '#e6e1cf' : '#0b0b0b'},
          labelFont: 'inherit',
          labelSize: 17,
          defaultDrawNodeHover: (context, node, settings) => {
            drawNodeHover(context, node, settings, dark);
          },
          nodeReducer: (node, attrs) => {
            if (state.hovered == null) {
              return attrs;
            }
            if (node === state.hovered || graph.areNeighbors(node, state.hovered)) {
              return {...attrs, zIndex: 1};
            }
            return {...attrs, color: dimmed, label: '', zIndex: 0};
          },
          edgeReducer: (edge, attrs) => {
            if (state.hovered == null) {
              return attrs;
            }
            if (graph.hasExtremity(edge, state.hovered)) {
              return {...attrs, color: highlight, zIndex: 1};
            }
            return {...attrs, hidden: true};
          },
        });
        sigma.on('enterNode', ({node}) => {
          state.hovered = node;
          setHovered(node);
          sigma.refresh();
        });
        sigma.on('leaveNode', () => {
          state.hovered = undefined;
          setHovered(null);
          sigma.refresh();
        });
        renderer = sigma;
      },
    );

    return () => {
      disposed = true;
      renderer?.kill();
    };
  }, [colorMode]);

  const hoveredNode =
    hovered == null ? null : data.nodes.find((n) => n.id === hovered) ?? null;

  return (
    <figure className={styles.figure}>
      <div className={styles.legend}>
        {data.groups.map((g) => (
          <span key={g} className={styles.legendItem}>
            <span
              className={styles.swatch}
              style={{backgroundColor: color(g, colorMode)}}
            />
            {g}
          </span>
        ))}
      </div>
      <div ref={containerRef} className={styles.canvas} />
      <figcaption className={styles.caption}>
        {hoveredNode ? (
          <>
            <strong>{hoveredNode.title}</strong> ({hoveredNode.year}) — degree{' '}
            {hoveredNode.degree} in this ego network
          </>
        ) : (
          <>
            Ego network of <em>Attention Is All You Need</em> — 133 papers, 393
            citation edges. Positions from <code>cugraph_force_atlas2</code>,
            colors from <code>cugraph_louvain</code>. Hover a node; drag or
            scroll to pan and zoom.
          </>
        )}
      </figcaption>
    </figure>
  );
}

export default function CitationEgoGraph(): React.ReactElement {
  return (
    <BrowserOnly fallback={<div style={{minHeight: 460}} />}>
      {() => <EgoCanvas />}
    </BrowserOnly>
  );
}
