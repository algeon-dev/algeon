import React, {useId, useMemo, useState} from 'react';
import styles from './MarketCollections.module.css';

type MarketRow = {
  vertex: number;
  community: number;
  community_size: number;
  representative_rank: number;
  product_id: string;
  brand_name: string | null;
  category: string;
  leaf_category: string | null;
  viewers: number;
  pagerank: number;
};

export type MarketCollectionCapture = {
  query: {query_id: string; rows: number};
  records: MarketRow[];
};

type CommunityRows = {
  id: number;
  size: number;
  markets: MarketRow[];
};

export default function MarketCollections({
  data,
  initialProductId,
}: {
  data: MarketCollectionCapture;
  // Leiden community numbers change between runs, so the initial view is
  // chosen by a market that belongs to it rather than by its number.
  initialProductId?: string;
}): React.ReactElement {
  const id = useId();
  const [selectedId, setSelectedId] = useState<number | null>(
    () => data.records.find(row => row.product_id === initialProductId)?.community ?? null,
  );
  const communities = useMemo(() => {
    const byId = new Map<number, MarketRow[]>();
    for (const row of data.records) {
      const markets = byId.get(row.community) ?? [];
      markets.push(row);
      byId.set(row.community, markets);
    }
    return Array.from(byId, ([communityId, markets]) => ({
      id: communityId,
      size: markets[0]?.community_size ?? 0,
      markets: markets.sort((left, right) => left.representative_rank - right.representative_rank),
    })).sort((left, right) => left.id - right.id);
  }, [data.records]);
  const selected = communities.find(community => community.id === selectedId) ?? communities[0];

  return (
    <figure className={styles.figure}>
      <div className={styles.controls}>
        <label htmlFor={`${id}-community`}>
          Captured Leiden community
          <select
            id={`${id}-community`}
            value={String(selected?.id ?? '')}
            onChange={event => setSelectedId(Number(event.target.value))}>
            {communities.map(community => (
              <option key={community.id} value={String(community.id)}>
                Community {community.id}: {community.size.toLocaleString()} markets
              </option>
            ))}
          </select>
        </label>
        <span className={styles.captureCount}>
          {communities.length} communities, {data.query.rows.toLocaleString()} representatives
        </span>
      </div>

      {selected ? (
        <div className={styles.results} aria-live="polite">
          <div className={styles.groupHeading}>
            <h4>Community {selected.id}</h4>
            <span>{selected.size.toLocaleString()} grouped markets</span>
          </div>
          <p className={styles.description}>
            Five market groups ranked by weighted PageRank within this captured community.
          </p>
          <div className={styles.tableScroll}>
            <table>
              <caption className={styles.visuallyHidden}>
                Five representative market groups in community {selected.id}
              </caption>
              <thead>
                <tr>
                  <th scope="col">Rank</th>
                  <th scope="col">Product group</th>
                  <th scope="col">Brand</th>
                  <th scope="col">Category</th>
                  <th scope="col">Leaf category</th>
                  <th scope="col">Viewers</th>
                  <th scope="col">PageRank</th>
                </tr>
              </thead>
              <tbody>
                {selected.markets.map(market => (
                  <tr key={market.vertex}>
                    <td>{market.representative_rank}</td>
                    <td><code>{market.product_id}</code></td>
                    <td>{market.brand_name ?? 'Brand not listed'}</td>
                    <td>{market.category}</td>
                    <td>{market.leaf_category ?? 'Leaf category not listed'}</td>
                    <td>{market.viewers.toLocaleString()}</td>
                    <td>{market.pagerank.toFixed(6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : <p>No market rows are available in this capture.</p>}

      <figcaption>
        The dropdown contains all {communities.length} communities in this query result. Each
        option shows the full grouped-market count; the table shows its five highest-ranked
        representatives.
      </figcaption>
    </figure>
  );
}
