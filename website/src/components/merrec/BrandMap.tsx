import React, {useId, useMemo, useState} from 'react';
import styles from './BrandMap.module.css';

type BrandRow = {
  id: number;
  name: string;
  fans: number;
  world: number;
  x: number;
  y: number;
};

type BrandWorld = {
  world: number;
  label: string;
  brands: number;
  fans: number;
};

export type BrandMapCapture = {
  query: {query_id: string; sql_file: string; rows: number};
  worlds: BrandWorld[];
  brands: BrandRow[];
};

type Dot = {brand: BrandRow; cx: number; cy: number; r: number};

const WORLD_CLASSES = [
  styles.world0, styles.world1, styles.world2, styles.world3, styles.world4,
  styles.world5, styles.world6, styles.world7, styles.world8, styles.world9,
  styles.world10, styles.world11, styles.world12, styles.world13,
];

// ForceAtlas2 with gravity yields a roughly round cloud, so a square plot wastes the least space.
const SIZE = 600;
const PADDING = 12;
const HIT_SLACK = 5;
const TOP_COUNT = 10;
const UNLABELLED = 'Unlabelled world';

// A fixed locale keeps server-rendered and hydrated text identical.
const integer = new Intl.NumberFormat('en-US');
const round = (value: number, digits: number) => Number(value.toFixed(digits));

function circles(dots: Dot[]): React.ReactElement[] {
  return dots.map(dot => <circle key={dot.brand.id} cx={dot.cx} cy={dot.cy} r={dot.r} />);
}

const worldClass = (index: number) => WORLD_CLASSES[index % WORLD_CLASSES.length];

export default function BrandMap({data}: {data: BrandMapCapture}): React.ReactElement {
  const id = useId();
  // Indexes into data.worlds; an empty set shows every world at full strength.
  const [selected, setSelected] = useState<ReadonlySet<number>>(() => new Set());
  const [hoverId, setHoverId] = useState<number | null>(null);

  const layout = useMemo(() => {
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (const brand of data.brands) {
      x0 = Math.min(x0, brand.x);
      x1 = Math.max(x1, brand.x);
      y0 = Math.min(y0, brand.y);
      y1 = Math.max(y1, brand.y);
    }
    const dx = x1 - x0 || 1;
    const dy = y1 - y0 || 1;
    // One scale for both axes keeps ForceAtlas2 distances comparable in x and y.
    const scale = (SIZE - 2 * PADDING) / Math.max(dx, dy);
    const left = (SIZE - dx * scale) / 2;
    const top = (SIZE - dy * scale) / 2;

    const worldIndex = new Map(data.worlds.map((world, index) => [world.world, index]));
    const dots: Dot[] = data.brands
      .map(brand => ({
        brand,
        cx: round(left + (brand.x - x0) * scale, 1),
        cy: round(top + (y1 - brand.y) * scale, 1),
        r: round(Math.min(9, Math.max(1.2, 1.2 + Math.sqrt(brand.fans) / 70)), 2),
      }))
      .sort((a, b) => b.brand.fans - a.brand.fans);

    const byWorld: Dot[][] = data.worlds.map(() => []);
    const unlabelled: Dot[] = [];
    for (const dot of dots) {
      const index = worldIndex.get(dot.brand.world);
      (index === undefined ? unlabelled : byWorld[index]).push(dot);
    }
    return {
      dots,
      byWorld,
      unlabelled,
      worldIndex,
      byId: new Map(dots.map(dot => [dot.brand.id, dot])),
      base: (
        <>
          <g className={`${styles.dots} ${styles.unlabelled}`}>{circles(unlabelled)}</g>
          {byWorld.map((group, index) => (
            <g key={data.worlds[index].world}
              className={`${styles.dots} ${styles.labelled} ${worldClass(index)}`}>
              {circles(group)}
            </g>
          ))}
        </>
      ),
    };
  }, [data.brands, data.worlds]);

  // Legend order, so highlight layers and swatches do not depend on click order.
  const selection = useMemo(
    () => data.worlds.flatMap((_, index) => (selected.has(index) ? [index] : [])),
    [data.worlds, selected],
  );
  const filtering = selection.length > 0;
  const selectedDots = useMemo(
    () => selection.flatMap(index => layout.byWorld[index]).sort((a, b) => b.brand.fans - a.brand.fans),
    [selection, layout.byWorld],
  );
  const topBrands = selectedDots.slice(0, TOP_COUNT);
  const selectedBrands = selection.reduce((sum, index) => sum + data.worlds[index].brands, 0);
  const selectedLinks = selection.reduce((sum, index) => sum + data.worlds[index].fans, 0);
  const hovered = hoverId === null ? undefined : layout.byId.get(hoverId);
  const hoveredIndex = hovered ? layout.worldIndex.get(hovered.brand.world) : undefined;
  const hoveredWorld = hoveredIndex === undefined ? null : data.worlds[hoveredIndex];

  const locate = (event: React.PointerEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width === 0) return;
    const ratio = SIZE / rect.width;
    const px = (event.clientX - rect.left) * ratio;
    const py = (event.clientY - rect.top) * ratio;
    let nearest: number | null = null;
    let nearestGap = HIT_SLACK;
    for (const dot of filtering ? selectedDots : layout.dots) {
      const gap = Math.hypot(dot.cx - px, dot.cy - py) - dot.r;
      if (gap < nearestGap) {
        nearestGap = gap;
        nearest = dot.brand.id;
      }
    }
    setHoverId(nearest);
  };
  const toggleWorld = (index: number) => {
    setSelected(current => {
      const next = new Set(current);
      if (!next.delete(index)) next.add(index);
      return next;
    });
    setHoverId(null);
  };
  const clearWorlds = () => {
    setSelected(new Set());
    setHoverId(null);
  };
  const brandCount = integer.format(data.brands.length);
  const panelTitle = selection.length === 1
    ? data.worlds[selection[0]].label
    : `${selection.length} worlds`;

  return (
    <figure className={styles.figure}>
      <div className={styles.stage}>
        <div className={styles.mapBox}>
          <svg className={styles.plot} viewBox={`0 0 ${SIZE} ${SIZE}`} role="img"
            aria-label={`Scatter plot of ${brandCount} brands. Each dot is a brand, sized by fans and colored by shopping world. Use the world buttons and the brand list for the same information as text.`}
            onPointerMove={locate}
            onPointerDown={locate}
            onPointerLeave={event => { if (event.pointerType !== 'touch') setHoverId(null); }}>
            <g className={filtering ? `${styles.layer} ${styles.faded}` : styles.layer}>{layout.base}</g>
            {selection.map(index => (
              <g key={data.worlds[index].world}
                className={`${styles.dots} ${styles.labelled} ${styles.highlight} ${worldClass(index)}`}>
                {circles(layout.byWorld[index])}
              </g>
            ))}
            {hovered ? <circle className={styles.ring} cx={hovered.cx} cy={hovered.cy} r={hovered.r + 2.5} /> : null}
          </svg>
          <p className={hovered ? styles.readout : `${styles.readout} ${styles.hint}`} aria-live="polite">
            {hovered
              ? `${hovered.brand.name}, ${hoveredWorld?.label ?? UNLABELLED}, ${integer.format(hovered.brand.fans)} fans`
              : 'Hover or tap a dot for its brand, world and fans.'}
          </p>
        </div>

        <div className={styles.side}>
          <div className={styles.sideInner}>
            <div className={styles.legendBar}>
              <span id={`${id}-worlds`} className={styles.legendTitle}>Shopping worlds</span>
              <span className={styles.legendStatus} aria-live="polite">
                {filtering ? `${selection.length} of ${data.worlds.length} selected` : 'Select to compare'}
              </span>
              <button type="button" className={styles.clear} onClick={clearWorlds} disabled={!filtering}>
                Clear
              </button>
            </div>
            <ul className={filtering ? `${styles.legend} ${styles.filtering}` : styles.legend}
              role="group" aria-labelledby={`${id}-worlds`}>
              {data.worlds.map((world, index) => (
                <li key={world.world}>
                  <button type="button" className={worldClass(index)} aria-pressed={selected.has(index)}
                    title={world.label} onClick={() => toggleWorld(index)}>
                    <i className={styles.swatch} />
                    <span>{world.label}</span>
                    <small>{integer.format(world.brands)}</small>
                  </button>
                </li>
              ))}
              {layout.unlabelled.length > 0 ? (
                <li className={styles.otherWorlds}>
                  <i className={`${styles.swatch} ${styles.unlabelled}`} />
                  <span>Unlabelled worlds</span>
                  <small>{integer.format(layout.unlabelled.length)}</small>
                </li>
              ) : null}
            </ul>
          </div>
        </div>
      </div>

      {filtering ? (
        <div className={styles.panel}>
          <div className={styles.panelHeading}>
            <span className={styles.swatches}>
              {selection.map(index => (
                <i key={data.worlds[index].world} className={`${styles.swatch} ${worldClass(index)}`} />
              ))}
            </span>
            <h4>{`${panelTitle}: top ${topBrands.length} brands by fans`}</h4>
            <span>{`${integer.format(selectedBrands)} brands, ${integer.format(selectedLinks)} brand–fan links selected`}</span>
          </div>
          <ol className={styles.topList}>
            {topBrands.map(dot => {
              const index = layout.worldIndex.get(dot.brand.world);
              return (
                <li key={dot.brand.id}>
                  <button type="button" className={index === undefined ? undefined : worldClass(index)}
                    onMouseEnter={() => setHoverId(dot.brand.id)}
                    onMouseLeave={() => setHoverId(null)}
                    onFocus={() => setHoverId(dot.brand.id)}
                    onBlur={() => setHoverId(null)}
                    onClick={() => setHoverId(dot.brand.id)}>
                    <i className={styles.swatch} />
                    <span>{dot.brand.name}</span>
                    {/* Some world hues are close, so a multi-world ranking names each brand's world. */}
                    {selection.length > 1 && index !== undefined ? <small>{data.worlds[index].label}</small> : null}
                    <strong>{integer.format(dot.brand.fans)}</strong>
                  </button>
                </li>
              );
            })}
          </ol>
        </div>
      ) : null}

      <figcaption>
        {`${brandCount} brands placed by ForceAtlas2 from shared fans; colors are Leiden shopping worlds. Dot area grows with fans. Gray dots belong to worlds without a label. Select one or more worlds to highlight them and rank their most-liked brands together.`}
      </figcaption>
    </figure>
  );
}
