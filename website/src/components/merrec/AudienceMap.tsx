import React, {useId, useMemo, useState} from 'react';
import styles from './AudienceMap.module.css';

type AudiencePoint = {
  id: number;
  x: number;
  y: number;
  clusterId: number;
  probability: number;
  views: number;
  shares: number[];
};

type AudienceCluster = {
  clusterId: number;
  label: string;
  size: number;
  theme: string | null;
  dominantCategory: string;
  meanSquaredFeatureShare: number;
  meanSquaredFeatureShares: number[];
};

export type AudienceCapture = {
  rows: number;
  categories: {column: string; name: string}[];
  noiseRows: number;
  assignedRows: number;
  clusterCount: number;
  clusters: AudienceCluster[];
  trustworthiness: {value: number; nNeighbors: number; inputDimensions: number};
  points: AudiencePoint[];
};

const clusterClasses: Record<number, string> = {
  [-1]: styles.noise,
  0: styles.cluster0,
  1: styles.cluster1,
  2: styles.cluster2,
  3: styles.cluster3,
  4: styles.cluster4,
};

export default function AudienceMap({data}: {data: AudienceCapture}): React.ReactElement {
  const id = useId();
  const [clusterFilter, setClusterFilter] = useState('all');
  const [showNoise, setShowNoise] = useState(true);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [hoverId, setHoverId] = useState<number | null>(null);
  const visible = useMemo(
    () => data.points.filter(point => {
      if (!showNoise && point.clusterId === -1) return false;
      if (clusterFilter === 'noise') return point.clusterId === -1;
      if (clusterFilter !== 'all') return point.clusterId === Number(clusterFilter);
      return true;
    }),
    [clusterFilter, data.points, showNoise],
  );
  const visibleIds = useMemo(() => new Set(visible.map(point => point.id)), [visible]);
  const bounds = useMemo(() => {
    const xs = data.points.map(point => point.x);
    const ys = data.points.map(point => point.y);
    const x0 = Math.min(...xs);
    const y0 = Math.min(...ys);
    return {
      x0,
      y0,
      dx: Math.max(...xs) - x0 || 1,
      dy: Math.max(...ys) - y0 || 1,
    };
  }, [data.points]);
  const xPosition = (point: AudiencePoint) => 58 + (point.x - bounds.x0) / bounds.dx * 820;
  const yPosition = (point: AudiencePoint) => 420 - (point.y - bounds.y0) / bounds.dy * 380;
  const active = visible.find(point => point.id === hoverId)
    ?? visible.find(point => point.id === selectedId)
    ?? visible[0];
  const moveSelection = (step: number) => {
    if (!active || visible.length === 0) return;
    const current = visible.findIndex(point => point.id === active.id);
    setSelectedId(visible[(current + step + visible.length) % visible.length].id);
    setHoverId(null);
  };
  const topShares = active
    ? active.shares
        .map((share, index) => ({share, category: data.categories[index].name}))
        .sort((left, right) => right.share - left.share)
        .slice(0, 5)
    : [];

  return (
    <figure className={styles.figure}>
      <div className={styles.controls}>
        <label htmlFor={`${id}-cluster`}>
          HDBSCAN group
          <select id={`${id}-cluster`} value={clusterFilter} onChange={event => {
            setClusterFilter(event.target.value);
            setHoverId(null);
          }}>
            <option value="all">All users</option>
            <option value="noise">Unassigned noise</option>
            {data.clusters.filter(cluster => cluster.clusterId >= 0).map(cluster => (
              <option key={cluster.clusterId} value={String(cluster.clusterId)}>
                Cluster {cluster.clusterId}: {cluster.theme}, {cluster.size.toLocaleString()} users
              </option>
            ))}
          </select>
        </label>
        <label className={styles.noiseToggle} htmlFor={`${id}-noise`}>
          <input id={`${id}-noise`} type="checkbox" checked={showNoise}
            onChange={event => { setShowNoise(event.target.checked); setHoverId(null); }} />
          Include unassigned users (noise)
        </label>
      </div>

      <div className={styles.legend}>
        {data.clusters.map(cluster => (
          <span key={cluster.clusterId}>
            <i className={clusterClasses[cluster.clusterId]} />
            {cluster.clusterId < 0 ? 'Noise' : `Cluster ${cluster.clusterId}, ${cluster.theme}`}
            <small>{cluster.size.toLocaleString()}</small>
          </span>
        ))}
        <span className={styles.visibleCount} aria-live="polite">
          {visible.length.toLocaleString()} of {data.rows.toLocaleString()} users shown
        </span>
      </div>

      <svg className={styles.plot} viewBox="0 0 900 470" role="img"
        aria-label="UMAP scatter plot colored by HDBSCAN group. Hover or select a point for its category shares. Use the previous and next buttons to browse points."
        onMouseLeave={() => setHoverId(null)}>
        <text x="468" y="461" className={styles.axisLabel}>UMAP dimension 0</text>
        <text x="14" y="236" className={styles.axisLabel} transform="rotate(-90 14 236)">
          UMAP dimension 1
        </text>
        {data.points.map(point => {
          const shown = visibleIds.has(point.id);
          const isActive = active?.id === point.id;
          return (
            <circle key={point.id} cx={xPosition(point)} cy={yPosition(point)}
              r={isActive ? 5 : 2.6}
            className={`${styles.point} ${clusterClasses[point.clusterId]} ${isActive ? styles.selected : ''}`}
              opacity={shown ? (point.clusterId === -1 ? 0.34 : 0.82) : 0.035}
              onMouseEnter={() => { if (shown) setHoverId(point.id); }}
              onClick={() => { if (shown) setSelectedId(point.id); }}>
              <title>{`User ${point.id}, ${point.clusterId < 0 ? 'noise' : `cluster ${point.clusterId}`}, membership strength ${point.probability.toFixed(3)}`}</title>
            </circle>
          );
        })}
      </svg>

      <div className={styles.navigation}>
        <button type="button" onClick={() => moveSelection(-1)} disabled={!active}>
          Previous user
        </button>
        <button type="button" onClick={() => moveSelection(1)} disabled={!active}>
          Next user
        </button>
        <span>These buttons and the point detail work with keyboard focus.</span>
      </div>

      <div className={styles.detail} aria-live="polite">
        {active ? <>
          <div className={styles.detailHeading}>
            <strong>Anonymous user {active.id.toLocaleString()}</strong>
            <span>{active.views.toLocaleString()} views</span>
            <span>{active.clusterId < 0 ? 'Unassigned noise' : `Cluster ${active.clusterId}`}</span>
          </div>
          <p className={styles.membership}>
            HDBSCAN membership strength: {active.probability.toFixed(3)}
          </p>
          <p className={styles.membership}>Top categories by share of view events</p>
          <ul className={styles.shares}>
            {topShares.map(({category, share}) => (
              <li key={category}>
                <span>{category}</span>
                <span className={styles.barTrack}>
                  <span className={styles.bar} style={{width: `${share * 100}%`}} />
                </span>
                <strong>{(share * 100).toFixed(1)}%</strong>
              </li>
            ))}
          </ul>
        </> : <p>No users match the selected filters.</p>}
      </div>

      <figcaption>
        UMAP used {data.trustworthiness.inputDimensions} square-root-transformed shares of item_view events
        to place {data.rows.toLocaleString()} users in two dimensions. HDBSCAN was fit separately to the
        original {data.trustworthiness.inputDimensions} dimensions. It marked {data.noiseRows.toLocaleString()}
        {' '}users as noise and assigned {data.assignedRows.toLocaleString()} users to five capture-local groups.
        Point distances do not estimate conversion. HDBSCAN probability represents membership strength.
      </figcaption>
    </figure>
  );
}
