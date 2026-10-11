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
  const [selected, setSelected] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (container == null) {
      return undefined;
    }
    let disposed = false;
    let renderer: {kill: () => void; redraw: () => void} | null = null;

    const start = () => Promise.all([import('graphology'), import('sigma')]).then(
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
        // Use solid hex values. Sigma's WebGL color packing does not take rgba() strings.
        const edgeColor = dark ? '#343b4a' : '#d9d8d3';
        const dimmed = dark ? '#1b1f29' : '#e4e3de';
        const highlight = dark ? '#e6b450' : '#52514e';
        // Mutable interaction state closed over by the reducers. The reducers run
        // during Sigma's constructor render, before the instance binding exists.
        const state: {
          selected: string | null;
          hovered: string | null;
        } = {
          selected: null,
          hovered: null,
        };
        const sigma = new Sigma(graph, container, {
          // Switching tabs hides the panel while physics frames still mutate
          // the graph, and each mutation schedules a Sigma render. Without
          // this setting that render throws on the zero-width container.
          allowInvalidContainer: true,
          renderLabels: false,
          defaultEdgeColor: edgeColor,
          labelColor: {color: dark ? '#e6e1cf' : '#0b0b0b'},
          labelFont: 'inherit',
          labelSize: 17,
          defaultDrawNodeHover: (context, node, settings) => {
            drawNodeHover(context, node, settings, dark);
          },
          nodeReducer: (node, attrs) => {
            const active = state.selected ?? state.hovered;
            if (active == null) {
              return attrs;
            }
            if (node === active) {
              return {...attrs, highlighted: true, zIndex: 1};
            }
            if (graph.areNeighbors(node, active)) {
              return {...attrs, zIndex: 1};
            }
            return {...attrs, color: dimmed, label: '', zIndex: 0};
          },
          edgeReducer: (edge, attrs) => {
            const active = state.selected ?? state.hovered;
            if (active == null) {
              return attrs;
            }
            if (graph.hasExtremity(edge, active)) {
              return {...attrs, color: highlight, zIndex: 1};
            }
            return {...attrs, hidden: true};
          },
        });
        // Viscoelastic jelly physics simulation
        interface PhysicsNode {
          id: string;
          x: number;
          y: number;
          homeX: number;
          homeY: number;
          vx: number;
          vy: number;
          fx: number;
          fy: number;
          mass: number;
          degree: number;
        }

        const physicsNodes = new Map<string, PhysicsNode>();
        for (const n of data.nodes) {
          const mass = 1.0 + 2.0 * Math.sqrt(n.degree / (maxDegree || 1));
          physicsNodes.set(n.id, {
            id: n.id,
            x: n.x,
            y: n.y,
            homeX: n.x,
            homeY: n.y,
            vx: 0,
            vy: 0,
            fx: 0,
            fy: 0,
            mass,
            degree: n.degree,
          });
        }

        const physicsEdges = data.edges.map(([src, dst]) => {
          const u = physicsNodes.get(src)!;
          const v = physicsNodes.get(dst)!;
          return {
            source: src,
            target: dst,
            restLength: Math.hypot(v.homeX - u.homeX, v.homeY - u.homeY),
          };
        });

        let draggedNodeId: string | null = null;
        let hasDragged = false;
        let dragStartPos: {x: number; y: number} | null = null;
        let animFrameId: number | null = null;

        const K_ANCHOR = 0.05;
        const K_EDGE = 0.015;
        const DAMPING = 0.91;

        // The physics advances in fixed steps so the motion does not depend on
        // the display refresh rate; one step per frame at 120 Hz was the tuned
        // feel, and a 60 Hz display would otherwise rebound at half speed.
        const STEP_MS = 1000 / 120;
        const MAX_FRAME_MS = 100;
        let lastFrameMs: number | null = null;
        let accumulatedMs = 0;

        const integrate = () => {
          // Soft-saturating restoring force: smooth non-linear jelly elasticity
          for (const p of physicsNodes.values()) {
            if (p.id === draggedNodeId) {
              p.fx = 0;
              p.fy = 0;
              continue;
            }
            const dx = p.x - p.homeX;
            const dy = p.y - p.homeY;
            const d = Math.hypot(dx, dy);
            const factor = K_ANCHOR / (1 + d / 20);
            p.fx = -factor * dx;
            p.fy = -factor * dy;
          }

          // Viscoelastic edge spring coupling with soft saturation and degree scaling
          for (const e of physicsEdges) {
            const u = physicsNodes.get(e.source);
            const v = physicsNodes.get(e.target);
            if (!u || !v) continue;
            const dx = v.x - u.x;
            const dy = v.y - u.y;
            const dist = Math.hypot(dx, dy) || 0.001;
            const displacement = dist - e.restLength;

            const maxDeg = Math.max(u.degree, v.degree, 1);
            const edgeFactor =
              K_EDGE / Math.sqrt(maxDeg) / (1 + Math.abs(displacement) / 15);
            const force = edgeFactor * displacement;

            const fx = (dx / dist) * force;
            const fy = (dy / dist) * force;

            if (u.id !== draggedNodeId) {
              u.fx += fx;
              u.fy += fy;
            }
            if (v.id !== draggedNodeId) {
              v.fx -= fx;
              v.fy -= fy;
            }
          }

          let maxVelocity = 0;
          let maxDisplacement = 0;

          // Mass-weighted acceleration and underdamped harmonic oscillation
          for (const p of physicsNodes.values()) {
            if (p.id === draggedNodeId) continue;

            const ax = p.fx / p.mass;
            const ay = p.fy / p.mass;
            p.vx = (p.vx + ax) * DAMPING;
            p.vy = (p.vy + ay) * DAMPING;

            p.x += p.vx;
            p.y += p.vy;

            graph.setNodeAttribute(p.id, 'x', p.x);
            graph.setNodeAttribute(p.id, 'y', p.y);

            const speed = Math.hypot(p.vx, p.vy);
            const dMag = Math.hypot(p.x - p.homeX, p.y - p.homeY);
            if (speed > maxVelocity) maxVelocity = speed;
            if (dMag > maxDisplacement) maxDisplacement = dMag;
          }

          return (
            draggedNodeId == null &&
            maxVelocity < 0.003 &&
            maxDisplacement < 0.015
          );
        };

        const stepSimulation = (frameMs: number) => {
          const elapsed = lastFrameMs == null ? STEP_MS : Math.min(frameMs - lastFrameMs, MAX_FRAME_MS);
          lastFrameMs = frameMs;
          accumulatedMs += elapsed;
          let settled = false;
          while (accumulatedMs >= STEP_MS && !settled) {
            settled = integrate();
            accumulatedMs -= STEP_MS;
          }
          sigma.refresh();

          if (settled) {
            for (const p of physicsNodes.values()) {
              p.x = p.homeX;
              p.y = p.homeY;
              p.vx = 0;
              p.vy = 0;
              graph.setNodeAttribute(p.id, 'x', p.homeX);
              graph.setNodeAttribute(p.id, 'y', p.homeY);
            }
            sigma.refresh();
            animFrameId = null;
            lastFrameMs = null;
            accumulatedMs = 0;
          } else {
            animFrameId = requestAnimationFrame(stepSimulation);
          }
        };

        const startSimulation = () => {
          if (animFrameId == null) {
            lastFrameMs = null;
            accumulatedMs = 0;
            animFrameId = requestAnimationFrame(stepSimulation);
          }
        };

        const triggerImpulse = (centerId: string, strength: number) => {
          const center = physicsNodes.get(centerId);
          if (!center) return;
          center.vx += (Math.random() - 0.5) * strength;
          center.vy += (Math.random() - 0.5) * strength;

          graph.forEachNeighbor(centerId, (neighborId) => {
            const n = physicsNodes.get(neighborId);
            if (!n) return;
            const dx = n.x - center.x;
            const dy = n.y - center.y;
            const dist = Math.hypot(dx, dy) || 1;
            n.vx += (dx / dist) * strength * 0.75;
            n.vy += (dy / dist) * strength * 0.75;
          });
          startSimulation();
        };

        // Entrance animation: gentle initial jelly ripple
        for (const p of physicsNodes.values()) {
          p.vx = (Math.random() - 0.5) * 0.4;
          p.vy = (Math.random() - 0.5) * 0.4;
        }
        startSimulation();

        sigma.on('downNode', ({node, event}) => {
          draggedNodeId = node;
          hasDragged = false;
          dragStartPos = {x: event.x, y: event.y};
          sigma.getCamera().enabledPanning = false;
        });

        const mouseCaptor = sigma.getMouseCaptor();
        const handlePointerMove = (e: {x: number; y: number}) => {
          if (!draggedNodeId) return;
          if (dragStartPos) {
            const dx = e.x - dragStartPos.x;
            const dy = e.y - dragStartPos.y;
            if (Math.hypot(dx, dy) > 3) {
              hasDragged = true;
            }
          }
          if (hasDragged) {
            container.style.cursor = 'grabbing';
            const pos = sigma.viewportToGraph(e);
            const p = physicsNodes.get(draggedNodeId);
            if (p) {
              p.vx = (pos.x - p.x) * 0.4;
              p.vy = (pos.y - p.y) * 0.4;
              p.x = pos.x;
              p.y = pos.y;
              graph.setNodeAttribute(draggedNodeId, 'x', pos.x);
              graph.setNodeAttribute(draggedNodeId, 'y', pos.y);
              startSimulation();
            }
          }
        };

        const handlePointerUp = () => {
          if (draggedNodeId) {
            draggedNodeId = null;
            container.style.cursor = state.hovered ? 'grab' : 'default';
            sigma.getCamera().enabledPanning = true;
            startSimulation();
          }
        };

        const handleTouchMove = (e: {touches: {x: number; y: number}[]}) => {
          if (e.touches && e.touches.length > 0) {
            handlePointerMove(e.touches[0]);
          }
        };

        mouseCaptor.on('mousemove', handlePointerMove);
        mouseCaptor.on('mouseup', handlePointerUp);

        const touchCaptor = sigma.getTouchCaptor();
        touchCaptor.on('touchmove', handleTouchMove);
        touchCaptor.on('touchup', handlePointerUp);

        window.addEventListener('mouseup', handlePointerUp);
        window.addEventListener('touchend', handlePointerUp);

        sigma.on('enterNode', ({node}) => {
          if (!draggedNodeId) {
            container.style.cursor = 'grab';
          }
          state.hovered = node;
          setHovered(node);
          triggerImpulse(node, 0.2);
          sigma.refresh();
        });
        sigma.on('leaveNode', () => {
          if (!draggedNodeId) {
            container.style.cursor = 'default';
          }
          state.hovered = null;
          setHovered(null);
          sigma.refresh();
        });
        sigma.on('clickNode', ({node}) => {
          if (hasDragged) {
            hasDragged = false;
            return;
          }
          state.selected = state.selected === node ? null : node;
          setSelected(state.selected);
          triggerImpulse(node, 0.8);
          sigma.refresh();
        });
        sigma.on('clickStage', () => {
          if (state.selected != null) {
            state.selected = null;
            setSelected(null);
            sigma.refresh();
          }
        });

        const handleDocumentClick = (event: MouseEvent) => {
          if (container && !container.contains(event.target as Node)) {
            if (state.selected != null) {
              state.selected = null;
              setSelected(null);
              sigma.refresh();
            }
          }
        };
        document.addEventListener('click', handleDocumentClick);

        renderer = {
          kill: () => {
            if (animFrameId != null) {
              cancelAnimationFrame(animFrameId);
            }
            mouseCaptor.off('mousemove', handlePointerMove);
            mouseCaptor.off('mouseup', handlePointerUp);
            touchCaptor.off('touchmove', handleTouchMove);
            touchCaptor.off('touchup', handlePointerUp);
            window.removeEventListener('mouseup', handlePointerUp);
            window.removeEventListener('touchend', handlePointerUp);
            document.removeEventListener('click', handleDocumentClick);
            sigma.kill();
          },
          redraw: () => sigma.refresh(),
        };
      },
    );

    // An inactive tab panel gives the container zero width. Start once the panel
    // is shown, and redraw each time it returns. Sigma resizes its canvas only on
    // window resize events, so it would otherwise keep its hidden 1x1 fallback.
    let started = false;
    const observer = new ResizeObserver(() => {
      if (container.clientWidth === 0) {
        return;
      }
      if (started) {
        renderer?.redraw();
      } else {
        started = true;
        start();
      }
    });
    observer.observe(container);

    return () => {
      disposed = true;
      observer.disconnect();
      renderer?.kill();
    };
  }, [colorMode]);

  const activeId = hovered ?? selected;
  const activeNode =
    activeId == null ? null : data.nodes.find((n) => n.id === activeId) ?? null;

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
        {activeNode ? (
          <>
            <strong>{activeNode.title}</strong> ({activeNode.year}), degree{' '}
            {activeNode.degree} in this ego network
          </>
        ) : (
          <>
            Ego network of <em>Attention Is All You Need</em> (133 papers, 393
            citation edges). Positions from <code>cugraph_force_atlas2</code>,
            colors from <code>cugraph_louvain</code>. Click, hover, or drag
            nodes to explore connections; click background to reset. Scroll to
            zoom.
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
