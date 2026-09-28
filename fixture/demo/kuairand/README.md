# KuaiRand

A sequential recommendation demo dataset with unbiased random intervention
splits. It contains user-video interaction logs with 12 feedback signals,
video metadata, video performance statistics, and user profile features.
It exercises multi-table hash joins, time-range pruning, high-cardinality
group-bys, and counterfactual policy evaluations on the native GPU path.

## Source

[KuaiRand](https://kuairand.com/) / [Zenodo record 10439422](https://zenodo.org/records/10439422).
The dataset comes from Kuaishou recommendation logs and provides three official
subsets:

| Variant | Upstream archive | Uncompressed | Users | Videos | Interactions |
| :--- | :--- | ---: | ---: | ---: | ---: |
| `pure` | `KuaiRand-Pure.tar.gz` (47 MB) | 194 MB | 27,285 | 7,583 | 2,622,668 |
| `1k` | `KuaiRand-1K.tar.gz` (1.1 GB) | 4.3 GB | 1,000 | 4,369,953 | 11,756,073 |
| `27k` | `KuaiRand-27K.tar.gz` (9.9 GB) | 46 GB | 27,285 | 32,038,725 | 323,464,444 |

Download via wget into `fixture/demo/kuairand/raw/`:

```bash
mkdir -p fixture/demo/kuairand/raw
cd fixture/demo/kuairand/raw
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar -xzvf KuaiRand-Pure.tar.gz
```

Alternatively, pass `--download` to `prepare.py` to fetch and extract the
selected variant automatically.

## Conversion

```bash
fixture/fixture.sh kuairand prepare --force
# same as: python3 fixture/demo/kuairand/scripts/prepare.py --variant pure --force
```

The converter requires the `duckdb` CLI and `pyarrow`. DuckDB parses the CSV
files, checks relational integrity invariants, builds surrogate sequence keys,
and sorts with external-memory spilling. PyArrow rewrites the sorted
intermediates with Snappy compression, format 2.6 page indexes, dictionary
encoding, selective min/max statistics, and bloom filters.

```
manifest.json                 source sizes and SHA-256, counts, per-table layout
parquet/
  users.parquet               27,285 rows, 1 row group, 1.14 MB
  videos.parquet              7,583 rows, 1 row group, 235 KB
  video_stats.parquet         7,583 rows, 1 row group, 2.56 MB
  interactions.parquet        2,622,668 rows, 3 row groups, 77.1 MB
```

The conversion fails without writing output if any of these invariants break:

- `user_id` is present and unique in `users`.
- `video_id` is present and unique in `videos` and `video_stats`.
- Every video in `video_stats` resolves to a row in `videos`.
- Every interaction in `interactions` has `user_id`, `video_id`, and `time_ms`.
- `is_rand` is 0 or 1.
- Binary feedback signals (`is_click`, `is_like`, `is_follow`, etc.) are 0 or 1.
- Every interaction resolves to existing `users` and `videos` keys.

## Tables

`users`, sorted by `user_id`:

| Column | Type | Derivation |
| :--- | :--- | :--- |
| `user_id` | Int32, not null | source key |
| `user_active_degree` | Utf8 | source activity level |
| `is_lowactive_period` | Int8 | source flag |
| `is_live_streamer`, `is_video_author` | Int8 | source author status (-124 for unknown) |
| `follow_user_num`, `fans_user_num`, `friend_user_num` | Int32 | source counts |
| `follow_user_num_range`, `fans_user_num_range`, `friend_user_num_range` | Utf8 | source count buckets |
| `register_days` | Int32 | source account age in days |
| `register_days_range` | Utf8 | source age bucket |
| `onehot_feat0` .. `onehot_feat17` | Int16 | encrypted user profile categorical feature ids |

`videos`, sorted by `video_id`:

| Column | Type | Derivation |
| :--- | :--- | :--- |
| `video_id` | Int32, not null | source key |
| `author_id` | Int64, not null | source author key |
| `video_type` | Utf8 | video format ("NORMAL" or "AD") |
| `upload_date` | Date32 | parsed `upload_dt` (`%Y-%m-%d`) |
| `upload_type` | Utf8 | source client upload type |
| `visible_status` | Int16 | source visibility state |
| `video_duration` | Int32 | video duration in milliseconds |
| `server_width`, `server_height` | Int16 | video resolution |
| `music_id` | Int64 | background music identifier |
| `music_type` | Int16 | background music category |
| `tag` | Utf8 | comma-separated category labels |

`video_stats`, sorted by `video_id`:

| Column | Type | Derivation |
| :--- | :--- | :--- |
| `video_id` | Int32, not null | joins `videos.video_id` |
| `counts` | Int64, not null | source observation sample size |
| `show_cnt`, `show_user_num` | Float64 | normalized exposure counts |
| `play_cnt`, `play_user_num`, `play_duration` | Float64 | playback engagement metrics |
| `complete_play_cnt`, `valid_play_cnt`, `long_time_play_cnt`, `short_time_play_cnt` | Float64 | playback completion milestones |
| `play_progress`, `comment_stay_duration` | Float64 | normalized progress and duration |
| `like_cnt`, `double_click_cnt`, `cancel_like_cnt` | Float64 | like interaction metrics |
| `comment_cnt`, `direct_comment_cnt`, `reply_comment_cnt`, `delete_comment_cnt` | Float64 | comment interaction metrics |
| `follow_cnt`, `share_cnt`, `download_cnt`, `collect_cnt`, `report_cnt` | Float64 | social actions and negative feedback |

`interactions`, sorted by `(interaction_time, interaction_id)`:

| Column | Type | Derivation |
| :--- | :--- | :--- |
| `interaction_id` | Int64, not null | deterministic dense row identifier |
| `interaction_time` | Timestamp(ms), not null | source epoch milliseconds stored as timezone-naive UTC wall time |
| `user_id` | Int32, not null | joins `users.user_id` |
| `video_id` | Int32, not null | joins `videos.video_id` |
| `date` | Date32, not null | parsed source date (`YYYYMMDD`) |
| `hourmin` | Int16, not null | interaction time of day (`HHMM`) |
| `time_ms` | Int64, not null | original millisecond timestamp |
| `is_click` | Int8, not null | click feedback signal (0 or 1) |
| `is_like` | Int8, not null | like button feedback (0 or 1) |
| `is_follow` | Int8, not null | follow author feedback (0 or 1) |
| `is_comment` | Int8, not null | comment feedback (0 or 1) |
| `is_forward` | Int8, not null | video share / forward feedback (0 or 1) |
| `is_hate` | Int8, not null | negative feedback signal (0 or 1) |
| `long_view` | Int8, not null | long viewing time indicator (0 or 1) |
| `play_time_ms`, `duration_ms` | Int64, not null | play time and item duration in milliseconds |
| `profile_stay_time`, `comment_stay_time` | Int64, not null | dwell times in profile and comments in milliseconds |
| `is_profile_enter` | Int8, not null | profile navigation indicator (0 or 1) |
| `is_rand` | Int8, not null | policy indicator: 1 for random exposure, 0 for standard feed |
| `tab` | Int8, not null | app surface context identifier (0 to 14) |

## Parquet layout for the GPU engine

- **Types.** Every column uses a type supported by the native query engine for
  filtering, grouping, and joining. Nested structures are flattened or omitted.
  `interaction_time` holds timezone-naive milliseconds in UTC wall time.
- **Integer keys.** `user_id` and `video_id` are dense integer keys, so joins
  and group-bys avoid string hashing.
- **Pruning.** Sorting `interactions` by `(interaction_time, interaction_id)`
  enables row-group min/max statistics to prune date and time ranges. `users`,
  `videos`, and `video_stats` are physically sorted by their integer primary key.
  Bloom filters cover `user_id` and `video_id` on `interactions`, `user_id` on
  `users`, and `video_id`, `author_id`, and `music_id` on `videos`.
- **Encoding.** Snappy compression, format 2.6, and page indexes throughout.
  Categorical columns with low cardinality use dictionary encoding. The 50
  float columns in `video_stats` and 18 encrypted feature columns in `users`
  omit min/max statistics to prevent metadata bloat.
- **Row groups.** 1,048,576 rows per row group for `interactions`; 131,072 rows
  for `users`, `videos`, and `video_stats`.
- **Ordering.** As in other fixtures, files omit `sorting_columns` metadata.
  Physical sorting serves min/max pruning rather than an output contract.

## Sample queries

Unbiased policy comparison (random exposure vs standard recommendation):

```sql
SELECT
    is_rand,
    count(*) AS interactions,
    round(avg(is_click), 4) AS ctr,
    round(avg(is_like), 4) AS like_rate,
    round(avg(is_forward), 4) AS forward_rate,
    round(avg(play_time_ms), 1) AS avg_play_ms
FROM interactions
GROUP BY is_rand
ORDER BY is_rand;
```

Three-way join across interactions, users, and video catalog:

```sql
SELECT
    v.video_type,
    u.user_active_degree,
    count(*) AS interaction_count,
    sum(i.is_like) AS total_likes,
    round(avg(i.play_time_ms), 1) AS avg_play_time
FROM interactions i
JOIN users u ON i.user_id = u.user_id
JOIN videos v ON i.video_id = v.video_id
WHERE i.interaction_time >= TIMESTAMP '2022-04-25 00:00:00'
GROUP BY v.video_type, u.user_active_degree
ORDER BY interaction_count DESC;
```

Catalog engagement aggregation joining videos and aggregate statistics:

```sql
SELECT
    v.video_type,
    v.upload_type,
    count(*) AS video_count,
    round(avg(s.play_progress), 4) AS avg_progress,
    round(avg(s.like_cnt), 2) AS avg_likes
FROM videos v
JOIN video_stats s ON v.video_id = s.video_id
WHERE v.upload_date >= DATE '2022-01-01'
GROUP BY v.video_type, v.upload_type
ORDER BY video_count DESC;
```
