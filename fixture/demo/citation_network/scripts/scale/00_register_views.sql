CREATE OR REPLACE VIEW edges_pre2000 AS
SELECT e.src, e.dst
FROM citation_edges e
JOIN papers ps ON ps.paper_id = e.src
JOIN papers pd ON pd.paper_id = e.dst
WHERE ps.year BETWEEN 1901 AND 2000 AND pd.year BETWEEN 1901 AND 2000;

CREATE OR REPLACE VIEW ai_nodes AS
SELECT paper_id FROM papers
WHERE year >= 2010 AND primary_fos IN (
  'Deep learning', 'Artificial neural network', 'Convolutional neural network',
  'Recurrent neural network', 'Natural language processing',
  'Reinforcement learning', 'Image segmentation', 'Feature extraction',
  'Object detection', 'Speech recognition');

CREATE OR REPLACE VIEW ai_edges AS
SELECT e.src, e.dst
FROM citation_edges e
JOIN ai_nodes a ON a.paper_id = e.src
JOIN ai_nodes b ON b.paper_id = e.dst;

CREATE OR REPLACE VIEW attention_seed AS
SELECT paper_id FROM papers
WHERE title = 'Attention is all you need' AND year = 2017;

CREATE OR REPLACE VIEW attention_ego_nodes AS
SELECT paper_id FROM (
  SELECT e.dst AS paper_id
  FROM citation_edges e JOIN attention_seed s ON s.paper_id = e.src
  UNION ALL
  SELECT src AS paper_id FROM (
    SELECT e.src, p.n_citation
    FROM citation_edges_by_dst e
    JOIN attention_seed s ON s.paper_id = e.dst
    JOIN papers p ON p.paper_id = e.src
    ORDER BY p.n_citation DESC, e.src LIMIT 110) t
  UNION ALL
  SELECT paper_id FROM attention_seed
) u GROUP BY paper_id;

CREATE OR REPLACE VIEW attention_ego_edges AS
SELECT e.src, e.dst
FROM citation_edges e
JOIN attention_ego_nodes a ON a.paper_id = e.src
JOIN attention_ego_nodes b ON b.paper_id = e.dst;

CREATE OR REPLACE VIEW similarity_candidates AS
SELECT paper_id, title, n_references
FROM papers
WHERE year BETWEEN 2015 AND 2020
  AND n_references >= 10
  AND primary_fos IN (
    'Natural language processing', 'Language model',
    'Machine translation', 'Question answering')
ORDER BY n_citation DESC, paper_id
LIMIT 12;

CREATE OR REPLACE VIEW bibliography_edges AS
SELECT e.src, -e.dst AS dst
FROM citation_edges e
JOIN similarity_candidates c ON c.paper_id = e.src;
