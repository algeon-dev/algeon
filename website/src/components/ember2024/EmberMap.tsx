import {useId, useMemo, useState} from 'react';
import type {CSSProperties, KeyboardEvent, MouseEvent, ReactElement} from 'react';

import styles from './EmberMap.module.css';

type ProjectionName = 'pca' | 'tsvd' | 'umap';

type EmberMapPoint = {
  id: number;
  sha256: string;
  family: string;
  weekId: number;
  clusterId: number;
  clusterProbability: number;
  pca: {x: number; y: number};
  tsvd: {x: number; y: number};
  umap: {x: number; y: number};
};

type EmberMapData = {
  rows: number;
  savedQueryIds: {
    pca: string;
    tsvd: string;
    umap: string;
    pca16_hdbscan: string;
  };
  points: EmberMapPoint[];
};

type Dot = {point: EmberMapPoint; cx: number; cy: number};

const PROJECTIONS: {value: ProjectionName; label: string}[] = [
  {value: 'umap', label: 'UMAP'},
  {value: 'pca', label: 'PCA'},
  {value: 'tsvd', label: 'Truncated SVD'},
];

const CLUSTER_COLORS = ['#2367a5', '#ce702b', '#178c66', '#a94b7c', '#7852a0', '#ba5143', '#557c38'];

const SIZE = 600;
const PADDING = 14;
const RADIUS = 3.6;
const HIT_SLACK = 4;

// A fixed locale keeps server-rendered and hydrated text identical.
const integer = new Intl.NumberFormat('en-US');
const round = (value: number) => Number(value.toFixed(1));

function clusterColor(clusterId: number): string {
  return clusterId < 0 ? '#777777' : CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];
}

function clusterLabel(clusterId: number): string {
  return clusterId === -1 ? 'Noise (-1)' : `Cluster ${clusterId}`;
}

function clusterStyle(clusterId: number): CSSProperties {
  return {'--cluster': clusterColor(clusterId)} as CSSProperties;
}

function circles(dots: Dot[]): ReactElement[] {
  return dots.map(dot => <circle key={dot.point.id} cx={dot.cx} cy={dot.cy} r={RADIUS} />);
}

export default function EmberMap({
  data,
  initialProjection = 'umap',
}: {
  data: EmberMapData;
  initialProjection?: ProjectionName;
}): ReactElement {
  const id = useId();
  const [projection, setProjection] = useState<ProjectionName>(initialProjection);
  const [search, setSearch] = useState('');
  // An empty set applies no cluster filter.
  const [selectedClusters, setSelectedClusters] = useState<ReadonlySet<number>>(() => new Set());
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [hoverId, setHoverId] = useState<number | null>(null);

  const clusters = useMemo(() => {
    const counts = new Map<number, number>();
    for (const point of data.points) counts.set(point.clusterId, (counts.get(point.clusterId) ?? 0) + 1);
    return [...counts].sort(([left], [right]) => left - right).map(([clusterId, count]) => ({clusterId, count}));
  }, [data.points]);

  const layout = useMemo(() => {
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    for (const point of data.points) {
      const {x, y} = point[projection];
      x0 = Math.min(x0, x);
      x1 = Math.max(x1, x);
      y0 = Math.min(y0, y);
      y1 = Math.max(y1, y);
    }
    const dx = x1 - x0 || 1;
    const dy = y1 - y0 || 1;
    // One scale for both axes keeps projected distances comparable in x and y.
    const scale = (SIZE - 2 * PADDING) / Math.max(dx, dy);
    const left = (SIZE - dx * scale) / 2;
    const top = (SIZE - dy * scale) / 2;

    const dots: Dot[] = data.points.map(point => ({
      point,
      cx: round(left + (point[projection].x - x0) * scale),
      cy: round(top + (y1 - point[projection].y) * scale),
    }));
    const byCluster = new Map<number, Dot[]>(clusters.map(({clusterId}) => [clusterId, []]));
    for (const dot of dots) byCluster.get(dot.point.clusterId)?.push(dot);
    return {
      byId: new Map(dots.map(dot => [dot.point.id, dot])),
      byCluster,
      // Noise comes first in cluster order, so assigned clusters paint above it.
      base: clusters.map(({clusterId}) => (
        <g key={clusterId} className={styles.dots} style={clusterStyle(clusterId)}>
          {circles(byCluster.get(clusterId) ?? [])}
        </g>
      )),
    };
  }, [clusters, data.points, projection]);

  const term = search.trim().toLocaleLowerCase();
  const filtering = selectedClusters.size > 0 || term !== '';
  const visible = useMemo(
    () => data.points.filter(point =>
      (selectedClusters.size === 0 || selectedClusters.has(point.clusterId)) &&
      (!term || point.sha256.toLocaleLowerCase().includes(term) || point.family.toLocaleLowerCase().includes(term))),
    [data.points, selectedClusters, term],
  );
  const visibleDots = useMemo(
    () => visible.flatMap(point => layout.byId.get(point.id) ?? []),
    [layout.byId, visible],
  );
  const highlight = useMemo(() => {
    if (!filtering) return null;
    const visibleIds = new Set(visible.map(point => point.id));
    return clusters.map(({clusterId}) => (
      <g key={clusterId} className={styles.dots} style={clusterStyle(clusterId)}>
        {circles((layout.byCluster.get(clusterId) ?? []).filter(dot => visibleIds.has(dot.point.id)))}
      </g>
    ));
  }, [clusters, filtering, layout.byCluster, visible]);

  const active = visible.find(point => point.id === (hoverId ?? selectedId)) ?? visible[0];
  const activeDot = active ? layout.byId.get(active.id) : undefined;
  const hovered = active !== undefined && active.id === hoverId;

  const nearest = (event: MouseEvent<SVGSVGElement>): number | null => {
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width === 0) return null;
    const ratio = SIZE / rect.width;
    const px = (event.clientX - rect.left) * ratio;
    const py = (event.clientY - rect.top) * ratio;
    let found: number | null = null;
    let foundGap = HIT_SLACK;
    for (const dot of visibleDots) {
      const gap = Math.hypot(dot.cx - px, dot.cy - py) - RADIUS;
      if (gap < foundGap) {
        foundGap = gap;
        found = dot.point.id;
      }
    }
    return found;
  };

  const selectNext = (step: number) => {
    if (!active || visible.length === 0) return;
    const current = visible.findIndex(point => point.id === active.id);
    setSelectedId(visible[(current + step + visible.length) % visible.length].id);
    setHoverId(null);
  };

  const handlePlotKeyDown = (event: KeyboardEvent<SVGSVGElement>) => {
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      selectNext(event.key === 'ArrowLeft' ? -1 : 1);
    } else if ((event.key === 'Enter' || event.key === ' ') && active) {
      event.preventDefault();
      setSelectedId(active.id);
      setHoverId(null);
    }
  };

  const toggleCluster = (clusterId: number) => {
    setSelectedClusters(current => {
      const next = new Set(current);
      if (!next.delete(clusterId)) next.add(clusterId);
      return next;
    });
    setHoverId(null);
  };

  const projectionLabel = PROJECTIONS.find(option => option.value === projection)?.label ?? projection;

  return (
    <figure className={styles.figure}>
      <div className={styles.stage}>
        <div className={styles.mapBox}>
          <svg className={hoverId === null ? styles.plot : `${styles.plot} ${styles.pointing}`}
            viewBox={`0 0 ${SIZE} ${SIZE}`} role="group" tabIndex={0}
            aria-label={`${projectionLabel} map of all ${integer.format(data.rows)} challenge Win32 files. Use left and right arrow keys to move through matching points, or Enter to select.`}
            onKeyDown={handlePlotKeyDown}
            onPointerMove={event => setHoverId(nearest(event))}
            onPointerDown={event => setHoverId(nearest(event))}
            onPointerLeave={event => { if (event.pointerType !== 'touch') setHoverId(null); }}
            onClick={event => {
              const found = nearest(event);
              if (found === null) return;
              setSelectedId(found);
              setHoverId(null);
            }}>
            <g className={filtering ? `${styles.layer} ${styles.faded}` : styles.layer}>{layout.base}</g>
            {highlight}
            {active && activeDot ? (
              <circle className={styles.active} style={clusterStyle(active.clusterId)}
                cx={activeDot.cx} cy={activeDot.cy} r={RADIUS + 2.4} />
            ) : null}
          </svg>
          <p className={hovered ? styles.readout : `${styles.readout} ${styles.hint}`}>
            {hovered && active
              ? `${active.family || 'Unknown family'}, ${clusterLabel(active.clusterId)}`
              : 'Hover or click a dot, or focus the map and press ← / →.'}
          </p>
        </div>

        <div className={styles.side}>
          <div className={styles.sideInner}>
            <div className={styles.segmented} role="group" aria-label="Projection">
              {PROJECTIONS.map(option => (
                <button key={option.value} type="button" aria-pressed={projection === option.value}
                  onClick={() => { setProjection(option.value); setHoverId(null); }}>
                  {option.label}
                </button>
              ))}
            </div>

            <input className={styles.search} type="search" value={search}
              aria-label="Search full SHA-256 or family" placeholder="Search SHA-256 or family"
              onChange={event => { setSearch(event.target.value); setHoverId(null); }} />

            <div className={styles.legendBar}>
              <span id={`${id}-clusters`} className={styles.legendTitle}>Clusters</span>
              <span className={styles.legendStatus} aria-live="polite">
                {selectedClusters.size > 0 ? `${selectedClusters.size} of ${clusters.length} selected` : null}
              </span>
              <button type="button" className={styles.clear} disabled={selectedClusters.size === 0}
                onClick={() => { setSelectedClusters(new Set()); setHoverId(null); }}>
                Clear
              </button>
            </div>
            <ul className={selectedClusters.size > 0 ? `${styles.legend} ${styles.filtering}` : styles.legend}
              role="group" aria-labelledby={`${id}-clusters`}>
              {clusters.map(({clusterId, count}) => (
                <li key={clusterId}>
                  <button type="button" style={clusterStyle(clusterId)}
                    aria-pressed={selectedClusters.has(clusterId)} onClick={() => toggleCluster(clusterId)}>
                    <i className={styles.swatch} />
                    <span>{clusterLabel(clusterId)}</span>
                    <small>{integer.format(count)}</small>
                  </button>
                </li>
              ))}
            </ul>
            <p className={styles.count} aria-live="polite">
              {`${integer.format(visible.length)} of ${integer.format(data.points.length)} points match`}
            </p>

            <div className={styles.detail} aria-live="polite">
              <div className={styles.detailBar}>
                {active ? (
                  <strong className={styles.family} style={clusterStyle(active.clusterId)}>
                    <i className={styles.swatch} />
                    <span>{active.family || 'Unknown family'}</span>
                  </strong>
                ) : <span className={styles.meta}>No points match.</span>}
                <button type="button" aria-label="Previous file" title="Previous file"
                  onClick={() => selectNext(-1)} disabled={!active}>‹</button>
                <button type="button" aria-label="Next file" title="Next file"
                  onClick={() => selectNext(1)} disabled={!active}>›</button>
              </div>
              {active ? <>
                <dl className={styles.facts}>
                  <div><dt>EMBER ID</dt><dd>{active.id}</dd></div>
                  <div><dt>Week</dt><dd>{active.weekId}</dd></div>
                  <div><dt>Cluster</dt><dd>{active.clusterId}</dd></div>
                  <div title="HDBSCAN membership strength"><dt>Membership</dt><dd>{active.clusterProbability.toFixed(3)}</dd></div>
                </dl>
                <code className={styles.sha}>{active.sha256}</code>
              </> : <p className={styles.meta}>Clear the SHA/family search or the cluster selection.</p>}
            </div>
          </div>
        </div>
      </div>
      <figcaption>
        All {integer.format(data.points.length)} saved challenge Win32 rows are included. Projection and cluster
        values come from saved SQL results; changing filters does not rerun SQL. Family is post-fit metadata.
        HDBSCAN clusters were fitted to PCA-16 coordinates, while this map can display PCA-2, TSVD-2, or UMAP-2.
        Cluster IDs are local to this capture. Coordinates are visual projections with no calibrated malware-risk
        interpretation. Noise (-1) records rows left unassigned under these HDBSCAN parameters; that label alone
        carries no novelty classification. <a href="/data/ember2024-map.json">Download the saved capture data</a>.
      </figcaption>
      <details className={styles.provenance}>
        <summary>Saved SQL capture IDs</summary>
        <ul>
          <li>PCA-2: <code>{data.savedQueryIds.pca}</code></li>
          <li>TSVD-2: <code>{data.savedQueryIds.tsvd}</code></li>
          <li>UMAP-2: <code>{data.savedQueryIds.umap}</code></li>
          <li>PCA-16 plus HDBSCAN: <code>{data.savedQueryIds.pca16_hdbscan}</code></li>
        </ul>
      </details>
    </figure>
  );
}
