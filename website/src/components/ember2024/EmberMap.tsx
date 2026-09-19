import {useId, useMemo, useState} from 'react';
import type {KeyboardEvent, ReactElement} from 'react';

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

const PROJECTIONS: {value: ProjectionName; label: string}[] = [
  {value: 'umap', label: 'UMAP'},
  {value: 'pca', label: 'PCA'},
  {value: 'tsvd', label: 'Truncated SVD'},
];

const CLUSTER_COLORS = ['#2367a5', '#ce702b', '#178c66', '#a94b7c', '#7852a0', '#ba5143', '#557c38'];

function clusterColor(clusterId: number): string {
  return clusterId < 0 ? '#777777' : CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];
}

export default function EmberMap({
  data,
  initialProjection = 'umap',
}: {
  data: EmberMapData;
  initialProjection?: ProjectionName;
}): ReactElement {
  const inputId = useId();
  const [projection, setProjection] = useState<ProjectionName>(initialProjection);
  const [search, setSearch] = useState('');
  const [cluster, setCluster] = useState('all');
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [hoverId, setHoverId] = useState<number | null>(null);

  const clusters = useMemo(
    () => [...new Set(data.points.map(point => point.clusterId))].sort((left, right) => left - right),
    [data.points],
  );
  const visible = useMemo(() => {
    const term = search.trim().toLocaleLowerCase();
    return data.points.filter(point =>
      (cluster === 'all' || point.clusterId === Number(cluster)) &&
      (!term || point.sha256.toLocaleLowerCase().includes(term) || point.family.toLocaleLowerCase().includes(term)));
  }, [cluster, data.points, search]);
  const visibleIds = useMemo(() => new Set(visible.map(point => point.id)), [visible]);
  const bounds = useMemo(() => {
    const first = data.points[0]?.[projection] ?? {x: 0, y: 0};
    return data.points.reduce(
      (result, point) => {
        const coordinate = point[projection];
        return {
          minX: Math.min(result.minX, coordinate.x),
          maxX: Math.max(result.maxX, coordinate.x),
          minY: Math.min(result.minY, coordinate.y),
          maxY: Math.max(result.maxY, coordinate.y),
        };
      },
      {minX: first.x, maxX: first.x, minY: first.y, maxY: first.y},
    );
  }, [data.points, projection]);
  const spanX = bounds.maxX - bounds.minX || 1;
  const spanY = bounds.maxY - bounds.minY || 1;
  const coordinateScale = Math.min(624 / spanX, 364 / spanY);
  const offsetX = 18 + (624 - spanX * coordinateScale) / 2;
  const offsetY = 18 + (364 - spanY * coordinateScale) / 2;
  const pointX = (point: EmberMapPoint) => offsetX + (point[projection].x - bounds.minX) * coordinateScale;
  const pointY = (point: EmberMapPoint) => 400 - offsetY - (point[projection].y - bounds.minY) * coordinateScale;
  const active = visible.find(point => point.id === (hoverId ?? selectedId)) ?? visible[0];

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

  return (
    <figure className={styles.figure}>
      <div className={styles.controls}>
        <label htmlFor={`${inputId}-projection`}>
          Projection
          <select id={`${inputId}-projection`} value={projection} onChange={event => {
            setProjection(event.target.value as ProjectionName);
            setHoverId(null);
          }}>
            {PROJECTIONS.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </label>
        <label htmlFor={`${inputId}-search`}>
          Search full SHA-256 or family
          <input id={`${inputId}-search`} type="search" value={search}
            placeholder="Try a SHA prefix or family name"
            onChange={event => { setSearch(event.target.value); setHoverId(null); }} />
        </label>
        <label htmlFor={`${inputId}-cluster`}>
          PCA-16 HDBSCAN cluster
          <select id={`${inputId}-cluster`} value={cluster} onChange={event => {
            setCluster(event.target.value);
            setHoverId(null);
          }}>
            <option value="all">All clusters</option>
            {clusters.map(clusterId => (
              <option key={clusterId} value={clusterId}>
                {clusterId === -1 ? 'Noise (-1)' : `Cluster ${clusterId}`}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className={styles.legend}>
        {clusters.map(clusterId => (
          <span key={clusterId}>
            <i style={{backgroundColor: clusterColor(clusterId)}} />
            {clusterId === -1 ? 'Noise (-1)' : `Cluster ${clusterId}`}
          </span>
        ))}
        <span className={styles.count} aria-live="polite">
          {visible.length.toLocaleString()} of {data.points.length.toLocaleString()} points match
        </span>
      </div>
      <svg className={styles.plot} viewBox="0 0 660 400" role="group" tabIndex={0}
        aria-label={`${projection.toUpperCase()} map of all ${data.rows.toLocaleString()} challenge Win32 files. Use left and right arrow keys to move through matching points, or Enter to select.`}
        onKeyDown={handlePlotKeyDown}
        onMouseLeave={() => setHoverId(null)}>
        {data.points.map(point => {
          const enabled = visibleIds.has(point.id);
          return <circle key={point.id} cx={pointX(point)} cy={pointY(point)}
            r={active?.id === point.id ? 5 : 2.7}
            fill={clusterColor(point.clusterId)}
            opacity={enabled ? 0.78 : 0.06}
            pointerEvents={enabled ? 'auto' : 'none'}
            stroke={active?.id === point.id ? 'var(--ifm-font-color-base)' : 'none'}
            strokeWidth={active?.id === point.id ? 1.5 : 0}
            onMouseEnter={() => { if (enabled) setHoverId(point.id); }}
            onClick={() => { if (enabled) { setSelectedId(point.id); setHoverId(null); } }}>
            <title>{`ID ${point.id}, ${point.family || 'unknown family'}, ${point.weekId}, cluster ${point.clusterId}`}</title>
          </circle>;
        })}
      </svg>
      <div className={styles.navigation}>
        <button type="button" onClick={() => selectNext(-1)} disabled={!active}>Previous file</button>
        <button type="button" onClick={() => selectNext(1)} disabled={!active}>Next file</button>
        <span>Focus the plot and use ← / → to navigate, Enter to select.</span>
      </div>
      <div className={styles.detail} aria-live="polite">
        {active ? <>
          <strong>{active.family || 'Unknown family'}</strong>
          <p className={styles.meta}>EMBER ID {active.id} · week {active.weekId} · capture-local cluster {active.clusterId} · HDBSCAN membership strength {active.clusterProbability.toFixed(3)}</p>
          <code className={styles.sha}>{active.sha256}</code>
        </> : <p>No points match. Clear the SHA/family search or choose another cluster.</p>}
      </div>
      <figcaption>
        All {data.points.length.toLocaleString()} saved challenge Win32 rows are included. Projection and cluster
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
