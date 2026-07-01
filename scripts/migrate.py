"""osu! mania dump -> PostgreSQL Migration"""

import os
import re
import glob
import tarfile
import argparse
import psycopg2
import psycopg2.extras
from pathlib import Path

from vsrg_irt import config


# Config
BATCH_SIZE = 5000


# Parsing INSERT VALUES
VALUE_RE = re.compile(r"'(?:[^'\\]|\\.)*'|NULL|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


# DDL
DDL = """
CREATE TABLE IF NOT EXISTS scores_mania (
    score_id       BIGINT      NOT NULL,
    user_id        INT         NOT NULL,
    beatmap_id     INT         NOT NULL,
    rank           VARCHAR(2)  NOT NULL,
    count300       INT         NOT NULL DEFAULT 0,
    count100       INT         NOT NULL DEFAULT 0,
    count50        INT         NOT NULL DEFAULT 0,
    countmiss      INT         NOT NULL DEFAULT 0,
    countgeki      INT         NOT NULL DEFAULT 0,
    countkatu      INT         NOT NULL DEFAULT 0,
    score          BIGINT      NOT NULL DEFAULT 0,
    accuracy       FLOAT       NOT NULL,
    enabled_mods   INT         NOT NULL DEFAULT 0,
    pp             FLOAT,
    date           TIMESTAMP   NOT NULL,
    in_top         BOOLEAN     NOT NULL DEFAULT FALSE,
    in_random      BOOLEAN     NOT NULL DEFAULT FALSE,
    PRIMARY KEY (score_id)
);

CREATE INDEX IF NOT EXISTS idx_scores_mania_user_beatmap
    ON scores_mania (user_id, beatmap_id);
CREATE INDEX IF NOT EXISTS idx_scores_mania_beatmap
    ON scores_mania (beatmap_id);

CREATE TABLE IF NOT EXISTS osu_beatmaps (
    beatmap_id       INT     PRIMARY KEY,
    beatmapset_id    INT,
    diff_size        FLOAT,
    difficultyrating FLOAT,
    playmode         SMALLINT,
    approved         SMALLINT,
    total_length     INT,
    playcount        INT,
    passcount        INT,
    bpm              FLOAT,
    version          TEXT
);

CREATE TABLE IF NOT EXISTS osu_user_stats_mania (
    user_id        INT     PRIMARY KEY,
    rank_score     FLOAT,
    rank           INT,
    accuracy_new   FLOAT,
    playcount      INT,
    level          FLOAT,
    x_rank_count   INT,
    xh_rank_count  INT,
    s_rank_count   INT,
    sh_rank_count  INT,
    a_rank_count   INT
);
"""


def calc_accuracy(c300, cgeki, c100, ckatu, c50, cmiss):
    total = cgeki + c300 + ckatu + c100 + c50 + cmiss
    if total == 0:
        return 0.0

    hits = 300 * cgeki + 300 * c300 + 200 * ckatu + 100 * c100 + 50 * c50
    return hits / (300 * total)


def parse_value_line (line):
    idx = line.upper().find("VALUES")
    if idx == -1:
        return []

    rest = line[idx + 6:].strip()

    rows = []
    depth = 0
    start = None
    i = 0
    in_string = False
    escape_next = False

    while i < len(rest):
        ch = rest[i]
        if escape_next:
            escape_next = False
        elif ch == '\\' and in_string:
            escape_next = True
        elif ch == "'" and not escape_next:
            in_string = not in_string
        elif not in_string:
            if ch == '(' and depth == 0:
                depth = 1
                start = i + 1
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0 and start is not None:
                    chunk = rest[start:i]
                    tokens = VALUE_RE.findall(chunk)
                    row = []
                    for t in tokens:
                        if t == 'NULL':
                            row.append(None)
                        elif t.startswith("'"):
                            row.append(t[1:-1].replace("\\'", "'").replace("\\\\", "\\"))
                        else:
                            try:
                                row.append(int(t))
                            except ValueError:
                                try:
                                    row.append(float(t))
                                except ValueError:
                                    row.append(t)

                    rows.append(row)
                    start = None

        i += 1

    return rows


# Transform columns

def safe_int(v, default=0):
    try:
        return int(v) if v is not None else default
    except(ValueError, TypeError):
        return default


def safe_float(v):
    try:
        return float(v) if v is not None else None
    except(ValueError, TypeError):
        return None


def convert_scores_mania_high(row, in_top, in_random):
    # score_id, beatmap_id, user_id, score, maxcombo, rank,
    # count50, count100, count300, countmiss, countgeki, countkatu,
    # perfect, enabled_mods, date, pp, replay, hidden, country_acronym
    if len(row) < 16:
        return None
    try:
        score_id     = safe_int(row[0])
        beatmap_id   = safe_int(row[1])
        user_id      = safe_int(row[2])
        score        = safe_int(row[3])
        rank         = str(row[5]) if row[5] is not None else ''
        count50      = safe_int(row[6])
        count100     = safe_int(row[7])
        count300     = safe_int(row[8])
        countmiss    = safe_int(row[9])
        countgeki    = safe_int(row[10])
        countkatu    = safe_int(row[11])
        enabled_mods = safe_int(row[13])
        date         = str(row[14]) if row[14] is not None else None
        pp           = safe_float(row[15])
        accuracy     = calc_accuracy(count300, countgeki, count100, countkatu, count50, countmiss)

        if date is None or score_id == 0:
            return None

        return (score_id, user_id, beatmap_id, rank,
                count300, count100, count50, countmiss, countgeki, countkatu,
                score, accuracy, enabled_mods, pp, date,
                in_top, in_random)
    except Exception:
        return None


def convert_beatmaps(row):
    # beatmap_id, beatmapset_id, user_id, filename, checksum, version,
    # total_length, hit_length, countTotal, countNormal, countSlider, countSpinner,
    # diff_drain, diff_size, diff_overall, diff_approach,
    # playmode, approved, last_update, difficultyrating, max_combo,
    # playcount, passcount, youtube_preview, score_version, osu_file_version,
    # deleted_at, bpm
    if len(row) < 28:
        return None
    try:
        return (
            safe_int(row[0]),
            safe_int(row[1]) if row[1] is not None else None,
            safe_float(row[13]),
            safe_float(row[19]),
            safe_int(row[16]),
            safe_int(row[17]),
            safe_int(row[6]),
            safe_int(row[21]),
            safe_int(row[22]),
            safe_float(row[27]),
            str(row[5]) if row[5] is not None else '',
        )
    except Exception:
        return None


def convert_user_stats(row):
    # user_id, count300, count100, count50, countMiss,
    # accuracy_total, accuracy_count, accuracy, playcount,
    # ranked_score, total_score,
    # x_rank_count, xh_rank_count, s_rank_count, sh_rank_count, a_rank_count,
    # rank, level, replay_popularity, fail_count, exit_count, max_combo,
    # country_acronym, rank_score, rank_score_index, accuracy_new,
    # last_update, last_played, total_seconds_played
    if len(row) < 26:
        return None
    try:
        return (
            safe_int(row[0]),
            safe_float(row[23]),
            safe_int(row[16]),
            safe_float(row[25]),
            safe_int(row[8]),
            safe_float(row[17]),
            safe_int(row[11]),
            safe_int(row[12]),
            safe_int(row[13]),
            safe_int(row[14]),
            safe_int(row[15]),
        )
    except Exception:
        return None


# UPSERT helper

def bulk_upsert_scores(cur, batch):
    psycopg2.extras.execute_values(cur, """
        INSERT INTO scores_mania
            (score_id, user_id, beatmap_id, rank,
             count300, count100, count50, countmiss, countgeki, countkatu,
             score, accuracy, enabled_mods, pp, date, in_top, in_random)
        VALUES %s
        ON CONFLICT (score_id) DO UPDATE SET
            in_top    = scores_mania.in_top    OR EXCLUDED.in_top,
            in_random = scores_mania.in_random OR EXCLUDED.in_random
    """, batch, page_size=BATCH_SIZE)


def bulk_upsert_generic(cur, table, cols, batch, pk=None):
    if pk is None:
        pk = [cols[0]]
    elif isinstance(pk, str):
        pk = [pk]

    col_str = ", ".join(cols)
    upd_cols = [c for c in cols if c not in pk]
    upd_str = ", ".join(f"{c} = EXCLUDED.{c}" for c in upd_cols)
    pk_str = ", ".join(pk)
    psycopg2.extras.execute_values(cur, f"""
        INSERT INTO {table} ({col_str})
        VALUES %s
        ON CONFLICT ({pk_str}) DO UPDATE SET {upd_str}
    """, batch, page_size=BATCH_SIZE)


# Single .sql file
TABLE_MAP = {
    "osu_scores_mania_high": "scores",
    "osu_beatmaps":          "beatmaps",
    "osu_user_stats_mania":  "users"
}

BEATMAPS_COLS = ["beatmap_id","beatmapset_id","diff_size","difficultyrating",
                 "playmode","approved","total_length","playcount","passcount","bpm","version"]
USERS_COLS    = ["user_id","rank_score","rank","accuracy_new","playcount",
                 "level","x_rank_count","xh_rank_count","s_rank_count","sh_rank_count","a_rank_count"]


def flush_batch(cur, key, batch):
    if not batch:
        return
    if key == "scores":
        bulk_upsert_scores(cur, batch)
    elif key == "beatmaps":
        bulk_upsert_generic(cur, "osu_beatmaps", BEATMAPS_COLS, batch)
    elif key == "users":
        bulk_upsert_generic(cur, "osu_user_stats_mania", USERS_COLS, batch)
    elif key == "difficulty":
        bulk_upsert_generic(cur, "osu_beatmap_difficulty", DIFF_COLS, batch,
                            pk=["beatmap_id","mode","mods"])

def process_sql_file(file_obj, in_top, in_random, cur):
    batches = {k: [] for k in TABLE_MAP.values()}
    counts  = {k: 0  for k in TABLE_MAP.values()}

    for raw in file_obj:
        line = raw.decode("utf-8", errors="replace").rstrip()

        if not line.startswith("INSERT INTO"):
            continue

        m = re.match(r"INSERT INTO `([^`]+)`", line)
        if not m:
            continue
        t_name = m.group(1)
        current_key = TABLE_MAP.get(t_name)
        if current_key is None:
            continue

        for row in parse_values_line(line):
            if current_key == "scores":
                converted = convert_scores_mania_high(row, in_top, in_random)
            elif current_key == "beatmaps":
                converted = convert_beatmaps(row)
            elif current_key == "users":
                converted = convert_user_stats(row)
            else:
                converted = None

            if converted is None:
                continue

            batches[current_key].append(converted)
            counts[current_key] += 1  # 추가

            if len(batches[current_key]) >= BATCH_SIZE:
                flush_batch(cur, current_key, batches[current_key])
                batches[current_key].clear()

    for key, batch in batches.items():
        flush_batch(cur, key, batch)

    for key, count in counts.items():
        if count > 0:
            print(f"    {key} 적재: {count:,}행")


# tar.bz2
TARGET_FILES = set(TABLE_MAP.keys())

def process_dump(tar_path, in_top, in_random, conn):
    print(f"\n[처리 중] {Path(tar_path).name}")
    with tarfile.open(tar_path, "r:bz2") as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".sql"):
                continue

            stem = Path(member.name).stem  # e.g. "osu_scores_mania_high"
            if stem not in TARGET_FILES:
                continue

            print(f"  → {member.name} ({member.size / 1024 / 1024:.1f} MB)")
            file_obj = tar.extractfile(member)
            if fileobj is None:
                continue
            with conn.cursor() as cur:
                process_sql_file(file_obj, in_top, in_random, cur)
            conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump-dir", required=True, help="folder of *.tar.bz2 dumps")
    args = ap.parse_args()

    conn = psycopg2.connect(config.DSN)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()
    print("테이블 생성 완료")

    pattern = os.path.join(args.dump_dir, "**", "*.tar.bz2")
    dump_files = sorted(glob.glob(pattern, recursive=True))

    if not dump_files:
        print(f"덤프 파일 없음: {pattern}")
        return

    for tar_path in dump_files:
        name = Path(tar_path).name.lower()
        in_top    = "top"    in name
        in_random = "random" in name
        try:
            process_dump(tar_path, in_top, in_random, conn)
        except Exception as e:
            print(f"  [오류] {tar_path}: {e}")
            conn.rollback()

    conn.close()
    print("\n완료!")


if __name__ == "__main__":
    main()
