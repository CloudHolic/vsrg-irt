-- Accuracy View
CREATE OR REPLACE VIEW v_irt_acc AS
WITH filtered AS (
    SELECT s.user_id, s.beatmap_id, s.accuracy, s.score, s.in_top, s.in_random,
           ROUND(m.diff_size)::int AS mania_keys,
           CASE WHEN (s.enabled_mods & 64)  <> 0 THEN 'DT'
                WHEN (s.enabled_mods & 256) <> 0 THEN 'HT'
                ELSE 'NM' END AS rate_group
    FROM scores_mania s
    JOIN osu_beatmaps m ON m.beatmap_id = s.beatmap_id
    WHERE (s.enabled_mods & 17261) = s.enabled_mods
      AND m.playmode = 3
),
     best AS (
         SELECT DISTINCT ON (f.user_id, f.beatmap_id, f.rate_group)
             f.user_id, f.beatmap_id, f.rate_group, f.accuracy, f.mania_keys,
             f.in_top, f.in_random
         FROM filtered f
         ORDER BY f.user_id, f.beatmap_id, f.rate_group, f.accuracy DESC, f.score DESC
     )
SELECT
    b.user_id, b.beatmap_id, b.rate_group,
    b.accuracy AS response,
    b.mania_keys,
    b.in_top, b.in_random
FROM best b;


-- Score View
CREATE OR REPLACE VIEW v_irt_score AS
WITH filtered AS (
    SELECT s.user_id, s.beatmap_id, s.score, s.accuracy, s.in_top, s.in_random,
           ROUND(m.diff_size)::int AS mania_keys,
           CASE WHEN (s.enabled_mods & 64)  <> 0 THEN 'DT'
                WHEN (s.enabled_mods & 256) <> 0 THEN 'HT'
                ELSE 'NM' END AS rate_group
    FROM scores_mania s
    JOIN osu_beatmaps m ON m.beatmap_id = s.beatmap_id
    WHERE (s.enabled_mods & 17260) = s.enabled_mods
      AND m.playmode = 3
),
     best AS (
         SELECT DISTINCT ON (f.user_id, f.beatmap_id, f.rate_group)
             f.user_id, f.beatmap_id, f.rate_group, f.score, f.mania_keys,
             f.in_top, f.in_random
         FROM filtered f
         ORDER BY f.user_id, f.beatmap_id, f.rate_group, f.score DESC, f.accuracy DESC
     )
SELECT
    b.user_id, b.beatmap_id, b.rate_group,
    (b.score / CASE b.rate_group WHEN 'HT' THEN 500000.0
                                 ELSE 1000000.0 END)::double precision AS response,
    b.mania_keys,
    b.in_top, b.in_random
FROM best b;