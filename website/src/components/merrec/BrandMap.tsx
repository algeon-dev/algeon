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

const WIDTH = 900;
const PADDING = 14;
const MIN_HEIGHT = 520;
const MAX_HEIGHT = 680;
const HIT_SLACK = 5;
const TOP_COUNT = 10;
const UNLABELLED = 'Unlabelled world';

// A fixed locale keeps server-rendered and hydrated text identical.
const integer = new Intl.NumberFormat('en-US');
const round = (value: number, digits: number) => Number(value.toFixed(digits));

function circles(dots: Dot[]): React.ReactElement[] {
  return dots.map(dot => <circle key={dot.brand.id} cx={dot.cx} cy={dot.cy} r={dot.r} />);
}

export default function BrandMap({data}: {data: BrandMapCapture}): React.ReactElement {
  const id = useId();
  const [selected, setSelected] = useState('all');
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
    const height = Math.round(Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, WIDTH * dy / dx)));
    // One scale for both axes keeps ForceAtlas2 distances comparable in x and y.
    const scale = Math.min((WIDTH - 2 * PADDING) / dx, (height - 2 * PADDING) / dy);
    const left = (WIDTH - dx * scale) / 2;
    const top = (height - dy * scale) / 2;

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
      height,
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
              className={`${styles.dots} ${styles.labelled} ${WORLD_CLASSES[index % WORLD_CLASSES.length]}`}>
              {circles(group)}
            </g>
          ))}
        </>
      ),
    };
  }, [data.brands, data.worlds]);

  const selectedIndex = selected === 'all'
    ? -1
    : data.worlds.findIndex(world => String(world.world) === selected);
  const selectedWorld = selectedIndex >= 0 ? data.worlds[selectedIndex] : null;
  const selectedClass = selectedIndex >= 0 ? WORLD_CLASSES[selectedIndex % WORLD_CLASSES.length] : '';
  const selectedDots = selectedIndex >= 0 ? layout.byWorld[selectedIndex] : [];
  const topBrands = selectedDots.slice(0, TOP_COUNT);
  const hovered = hoverId === null ? undefined : layout.byId.get(hoverId);
  const hoveredIndex = hovered ? layout.worldIndex.get(hovered.brand.world) : undefined;
  const hoveredWorld = hoveredIndex === undefined ? null : data.worlds[hoveredIndex];

  const locate = (event: React.PointerEvent<SVGSVGElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width === 0) return;
    const ratio = WIDTH / rect.width;
    const px = (event.clientX - rect.left) * ratio;
    const py = (event.clientY - rect.top) * ratio;
    let nearest: number | null = null;
    let nearestGap = HIT_SLACK;
    for (const dot of selectedIndex >= 0 ? selectedDots : layout.dots) {
      const gap = Math.hypot(dot.cx - px, dot.cy - py) - dot.r;
      if (gap < nearestGap) {
        nearestGap = gap;
        nearest = dot.brand.id;
      }
    }
    setHoverId(nearest);
  };
  const selectWorld = (value: string) => {
    setSelected(value);
    setHoverId(null);
  };
  const brandCount = integer.format(data.brands.length);

  return (
    <figure className={styles.figure}>
      <div className={styles.controls}>
        <label htmlFor={`${id}-world`}>
          Shopping world
          <select id={`${id}-world`} value={selected} onChange={event => selectWorld(event.target.value)}>
            <option value="all">All worlds</option>
            {data.worlds.map(world => (
              <option key={world.world} value={String(world.world)}>
                {`${world.label} (${integer.format(world.brands)} brands)`}
              </option>
            ))}
          </select>
        </label>
      </div>

      <svg className={styles.plot} viewBox={`0 0 ${WIDTH} ${layout.height}`} role="img"
        aria-label={`Scatter plot of ${brandCount} brands. Each dot is a brand, sized by fans and colored by shopping world. Use the world selector and the brand list below for the same information as text.`}
        onPointerMove={locate}
        onPointerDown={locate}
        onPointerLeave={event => { if (event.pointerType !== 'touch') setHoverId(null); }}>
        <g className={selectedWorld ? `${styles.layer} ${styles.faded}` : styles.layer}>{layout.base}</g>
        {selectedWorld ? (
          <g className={`${styles.dots} ${styles.labelled} ${styles.highlight} ${selectedClass}`}>
            {circles(selectedDots)}
          </g>
        ) : null}
        {hovered ? <circle className={styles.ring} cx={hovered.cx} cy={hovered.cy} r={hovered.r + 2.5} /> : null}
      </svg>

      <p className={styles.tooltip} aria-live="polite">
        {hovered
          ? `${hovered.brand.name}, ${hoveredWorld?.label ?? UNLABELLED}, ${integer.format(hovered.brand.fans)} fans`
          : 'Hover a dot to see its brand, world and fans.'}
      </p>

      {selectedWorld ? (
        <div className={`${styles.panel} ${selectedClass}`}>
          <div className={styles.panelHeading}>
            <i className={styles.swatch} />
            <h4>{`${selectedWorld.label}: top ${topBrands.length} brands by fans`}</h4>
            <span>{`${integer.format(selectedWorld.brands)} brands, ${integer.format(selectedWorld.fans)} fans in this world`}</span>
          </div>
          <ol className={styles.topList}>
            {topBrands.map(dot => (
              <li key={dot.brand.id}>
                <button type="button"
                  onMouseEnter={() => setHoverId(dot.brand.id)}
                  onMouseLeave={() => setHoverId(null)}
                  onFocus={() => setHoverId(dot.brand.id)}
                  onBlur={() => setHoverId(null)}
                  onClick={() => setHoverId(dot.brand.id)}>
                  <span>{dot.brand.name}</span>
                  <strong>{integer.format(dot.brand.fans)}</strong>
                </button>
              </li>
            ))}
          </ol>
        </div>
      ) : (
        <ul className={styles.legend}>
          {data.worlds.map((world, index) => (
            <li key={world.world}>
              <button type="button" onClick={() => selectWorld(String(world.world))}>
                <i className={`${styles.swatch} ${WORLD_CLASSES[index % WORLD_CLASSES.length]}`} />
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
      )}

      <figcaption>
        {`${brandCount} brands placed by ForceAtlas2 from shared fans; colors are Leiden shopping worlds. Dot area grows with fans. Gray dots belong to worlds without a label. Select a world to highlight it and list its most-liked brands.`}
      </figcaption>
    </figure>
  );
}
