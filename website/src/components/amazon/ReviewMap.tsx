import React, {useId, useMemo, useState} from 'react';
import styles from './styles.module.css';

type ReviewPoint = {
  id: number;
  x: number;
  y: number;
  year: number;
  rating: number;
  product: string;
  text: string;
};

export type ReviewMapData = {
  fittedRows: number;
  captured: string;
  points: ReviewPoint[];
};

export default function ReviewMap({data}: {data: ReviewMapData}): React.ReactElement {
  const inputId = useId();
  const [search, setSearch] = useState('');
  const [year, setYear] = useState('all');
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [hoverId, setHoverId] = useState<number | null>(null);
  const visible = useMemo(() => {
    const term = search.toLocaleLowerCase().trim();
    return data.points.filter(point =>
      (year === 'all' || point.year === Number(year)) &&
      (!term || `${point.product} ${point.text}`.toLocaleLowerCase().includes(term)));
  }, [data.points, search, year]);
  const visibleIds = useMemo(() => new Set(visible.map(point => point.id)), [visible]);
  const bounds = useMemo(() => {
    const xs = data.points.map(point => point.x);
    const ys = data.points.map(point => point.y);
    const x0 = Math.min(...xs);
    const y0 = Math.min(...ys);
    return {x0, y0, dx: Math.max(...xs) - x0 || 1, dy: Math.max(...ys) - y0 || 1};
  }, [data.points]);
  const pointX = (point: ReviewPoint) => 18 + (point.x - bounds.x0) / bounds.dx * 624;
  const pointY = (point: ReviewPoint) => 382 - (point.y - bounds.y0) / bounds.dy * 364;
  const active = visible.find(point => point.id === (hoverId ?? selectedId)) ?? visible[0];
  const selectNext = (step: number) => {
    if (!active || visible.length === 0) return;
    const current = visible.findIndex(point => point.id === active.id);
    setSelectedId(visible[(current + step + visible.length) % visible.length].id);
    setHoverId(null);
  };

  return (
    <figure className={styles.figure}>
      <div className={styles.controls}>
        <label htmlFor={`${inputId}-search`}>
          Find words in reviews or product names
          <input id={`${inputId}-search`} type="search" value={search}
            placeholder="Try charging, left, or Sony"
            onChange={event => { setSearch(event.target.value); setHoverId(null); }} />
        </label>
        <label htmlFor={`${inputId}-year`}>
          Review year
          <select id={`${inputId}-year`} value={year} onChange={event => {
            setYear(event.target.value); setHoverId(null);
          }}>
            <option value="all">2021 and 2022</option>
            <option value="2021">2021</option>
            <option value="2022">2022</option>
          </select>
        </label>
      </div>
      <div className={styles.legend}>
        <span><i className={styles.year2021} />2021</span>
        <span><i className={styles.year2022} />2022</span>
        <span aria-live="polite">{visible.length.toLocaleString()} of {data.points.length.toLocaleString()} displayed reviews match</span>
      </div>
      <svg className={styles.plot} viewBox="0 0 660 400" role="img"
        aria-label="UMAP projection of sampled headphone reviews. Select a point to read its review, or use the previous and next buttons below."
        onMouseLeave={() => setHoverId(null)}>
        {data.points.map(point => {
          const enabled = visibleIds.has(point.id);
          return <circle key={point.id} cx={pointX(point)} cy={pointY(point)}
            r={active?.id === point.id ? 5 : 2.8}
            className={point.year === 2021 ? styles.year2021 : styles.year2022}
            opacity={enabled ? 0.75 : 0.07}
            stroke={active?.id === point.id ? 'var(--ifm-font-color-base)' : 'none'}
            onMouseEnter={() => { if (enabled) setHoverId(point.id); }}
            onClick={() => { if (enabled) setSelectedId(point.id); }}>
            <title>{`${point.product}, ${point.year}, rating ${point.rating}/5`}</title>
          </circle>;
        })}
      </svg>
      <div className={styles.navigation}>
        <button type="button" onClick={() => selectNext(-1)} disabled={!active}>Previous review</button>
        <button type="button" onClick={() => selectNext(1)} disabled={!active}>Next review</button>
      </div>
      <div className={styles.detail}>
        {active ? <>
          <strong>{active.product}</strong>
          <p className={styles.meta}>Review {active.id} · {active.year} · Rating {active.rating}/5</p>
          <blockquote>{active.text}</blockquote>
        </> : <p>No displayed reviews match. Clear the text or year filter to browse again.</p>}
      </div>
      <figcaption>
        UMAP was fitted to {data.fittedRows.toLocaleString()} sampled low-rating headphone reviews.
        This view displays {data.points.length.toLocaleString()} evenly spaced review IDs from that result,
        captured {data.captured}. Filters highlight this saved sample; they do not refit UMAP.
        Nearby points are candidates for reading together. Map distance is not a defect score.
      </figcaption>
    </figure>
  );
}
