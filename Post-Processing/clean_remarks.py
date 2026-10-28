#!/usr/bin/env python3
"""
Conservative, non-destructive cleaning of the EC production-run remarks.

Input : ec_remarks.parquet (one row per scan; `output` is the raw VLM transcription)
Output: ec_remarks_clean.parquet  - every input row and column unchanged, plus:
            remarks_clean       kept lines, original order, one per distinct line; in forms that
                                finished normally, a line that occurred N>1 times ends with " [×N]"
                                (not shown for token-capped forms, where repeats are loop artifacts)
            has_remarks_clean   True if any line was kept
            n_lines_raw         non-empty lines in `output`
            n_lines_clean       lines in remarks_clean
            n_removed_<rule>    distinct lines removed by each rule (see below)
        ec_remark_lines.parquet - one row per distinct line per form, kept AND removed,
                                  with the rule that removed it (null = kept)

Nothing is deleted: `output` is left exactly as produced, and every removed line is
listed with its rule in ec_remark_lines.parquet.

Lines are compared after normalisation (lowercase, accents stripped, punctuation -> space).
Rules, applied to each distinct line of a form:
  repeat   : a line that recurs in the same form is kept once (first occurrence's text)
             and its count is shown as " [×N]". Not counted as a removal of content.
  symbols  : no letters or digits at all (stray marks, Vienna-congress weather symbols)
  reading  : only numbers plus statistics/compass words or single letters
             ("Max = 97.0 on 18th", "Mean 59", "NE 8", "Sept 14 a.m.")
  zero_count: a printed summary-table entry with a zero count ("Aurora 0", "Gales 0")
  null_entry: only "No", "None", "Nil", "Nothing"
  cut_off_repeat: in a token-capped output, the cut-off final line when it is the start of
             another line in the same form ("Qu" after "Quel Cool"); its content is in that line
  plain_weather: every word is plain weather-state, time, direction or function vocabulary
             ("Rain", "Cloudy", "Fine warm", "Light rain at night", "All day"), and the line
             has fewer than 8 words (longer plain-vocabulary narratives are kept).
             Lines are KEPT if they contain any other word, including intensifiers (very,
             heavy, strong, high, severe...), event words (storm, thunder, lightning, frost,
             hail, gale...), expressive words (beautiful, gloomy, sultry...), haze/smoke,
             thaw/freeze, month or season names, or anything misspelled/unrecognised.

Usage: python clean_remarks.py ec_remarks.parquet OUTDIR
Needs: duckdb, pyarrow, pandas
"""
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

def W(s):
    return frozenset(s.split())

# --- plain vocabulary: a line made only of these (plus numbers/ordinals/single letters) is removed
PLAIN = W("""
clear clearing cleared clears cloudy cloud clouds clouded clouding cldy cloudless overcast fine fair dull
bright sunny sunshine sun shining partly mostly good
rain rains rained raining rainy rainfall shower showers showery showering drizzle drizzling drizzly
sprinkle sprinkles sprinkling snow snows snowing snowed snowy snowfall flurries flurry snowflurries flakes
sleet sleety precipitation precip drops falling fell fall
fog foggy fogs mist misty
warm warmer cold colder cool cooler hot hotter mild milder chilly damp wet dry drier
wind winds windy breeze breezy breezes calm calms fresh gusty gust gusts blustery blowing drifting drift drifted
unsettled changeable variable
light lght lgt lt slight slightly little moderate mod gentle some few occasional occasionally intermittent
intermittently continuous steady quite rather fairly generally
day days night nights morning mornings morn forenoon afternoon afternoons aft evening evenings eve noon
midnight am pm early late later all during in at on of the and with to from till til until then after before
by through thru throughout part times time hours hrs hour hr today tonight since again still but also becoming
turning turned changing changed began commenced started ended stopped continued continuing about it is was
were be been has had this that or so as for up down off out here weather clock oclock half whole
do ditto dto
n s e w ne nw se sw nne ene ese sse ssw wsw wnw nnw north south east west northerly southerly easterly westerly
northeast northwest southeast southwest bc
pluie pluies neige neiges beau belle temps nuageux nuages nuageuse couvert vent vents froid froide chaud chaude
doux brume brouillard nuit jour journee matin soir apres midi la le les de du des et au aux dans tout toute
pendant leger legere humide sec clair frais gris averse averses bruine faible
""")

# --- statistics/compass vocabulary for the `reading` rule (numbers required or single letters only)
STAT = W("""
max min mean maximum minimum temp temperature total totals av avg average highest lowest range monthly month
precip precipitation rainfall snowfall
inches inch ins in on at of to days day deg degrees f trace traces measured gauge reading readings
am pm a m p bulb wet dry mmt
jan january feb february mar march apr april may jun june jul july aug august sep sept september oct october
nov november dec december
monday tuesday wednesday thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat tr
n s e w ne nw se sw nne ene ese sse ssw wsw wnw nnw north south east west c calm calms
""")

# a line made only of these is an empty entry ("None", "Nil", "No")
NULLS = W("no none nil nothing remarks remark")

# plain-vocabulary lines with at least this many words are kept: they read as observer-written
# weather narratives rather than recorded states
LONG_PLAIN_WORDS = 8

# labels that may precede/follow a zero in a summary table; one-word lines with a zero count are
# treated as table entries whatever the word (HTR often garbles printed labels: "Salone 0")
SUMMARY_LABELS = W("""
clear cloudy fog fogs aurora auroras gale gales thunder thunderstorm thunderstorms hail halo halos solar lunar
rain snow frost days calm calms storms storm lightning corona coronas parhelia rainbow rainbows mist dew
""")

NUM = re.compile(r"^\d+$")
ORD = re.compile(r"^\d+(st|nd|rd|th|a|p|am|pm|h|hr|hrs|in)?$")


def rule_for(norm: str):
    """Return the removal rule for a normalised line, or None to keep it."""
    toks = norm.split()
    if not toks:
        return "symbols"
    alpha = [t for t in toks if not ORD.match(t)]
    nums = [t for t in toks if NUM.match(t)]
    # reading: numbers/ordinals with only statistics, compass, month words or single letters
    if all(len(t) < 2 or t in STAT for t in alpha):
        return "reading"
    # zero_count: printed summary-table entry with a 0 count
    if nums and all(n == "0" for n in nums) and len(alpha) >= 1:
        words = [t for t in alpha if len(t) >= 2]
        if len(words) == 1 or (len(words) == 2 and any(t in SUMMARY_LABELS for t in words)):
            return "zero_count"
    # null_entry: "None", "Nil", "No"
    if all(t in NULLS or len(t) < 2 for t in alpha):
        return "null_entry"
    # plain_weather: every token is plain vocabulary, a number/ordinal, or a single letter
    if all(t in PLAIN or ORD.match(t) or len(t) < 2 for t in toks):
        if len([t for t in alpha if len(t) >= 2]) >= LONG_PLAIN_WORDS:
            return None
        return "plain_weather"
    return None


# cut_off_repeat is assigned in SQL: the last line of a token-capped output, cut mid-line,
# when it is the start of another line in the same form (the loop restarting)
RULES = ["symbols", "reading", "zero_count", "null_entry", "plain_weather", "cut_off_repeat"]


def main(src: str, outdir: str):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=true")
    con.execute("SET memory_limit='5GB'")
    con.execute(f"CREATE TABLE r AS SELECT row_number() OVER () - 1 AS _row, * FROM read_parquet('{src}')")

    # explode outputs of has_remarks rows into lines; normalise; group repeats within a form
    con.execute(r"""
        CREATE TABLE ln AS
        WITH x AS (
            SELECT _row, unnest(string_split(output, chr(10))) AS line,
                   generate_subscripts(string_split(output, chr(10)), 1) AS line_no
            FROM r WHERE has_remarks AND output IS NOT NULL)
        SELECT _row, line_no, trim(line) AS line,
               trim(regexp_replace(regexp_replace(strip_accents(lower(trim(line))), '[^a-z0-9\s]', ' ', 'g'),
                                   '\s+', ' ', 'g')) AS norm
        FROM x WHERE trim(line) <> ''""")
    con.execute("""
        CREATE TABLE ul AS
        SELECT _row, norm, min(line_no) AS first_line_no, arg_min(line, line_no) AS text, count(*) AS times_in_form
        FROM ln GROUP BY _row, norm""")

    # classify each distinct normalised string once
    norms = [n for (n,) in con.execute("SELECT DISTINCT norm FROM ul").fetchall()]
    cls = pd.DataFrame({"norm": norms, "removed_by": [rule_for(n) for n in norms]})
    con.register("cls", cls)
    # truncated outputs: the final line is cut off by the token cap. If it occurs once and is the
    # beginning of another line in the same form that is itself kept, it is a partial repeat of
    # that line, whose full text stays in remarks_clean.
    con.execute("""
        CREATE TABLE cut AS
        WITH last AS (
            SELECT ln._row, arg_max(ln.norm, ln.line_no) AS norm
            FROM ln JOIN r USING (_row) WHERE r.finish_reason = 'length' GROUP BY ln._row)
        SELECT u._row, u.norm FROM ul u JOIN last USING (_row, norm)
        WHERE u.times_in_form = 1 AND EXISTS (
            SELECT 1 FROM ul v JOIN cls c ON c.norm = v.norm
            WHERE v._row = u._row AND v.norm <> u.norm AND starts_with(v.norm, u.norm) AND c.removed_by IS NULL)""")
    con.execute("""
        CREATE TABLE ulc AS
        SELECT ul.*, CASE WHEN cut._row IS NOT NULL THEN 'cut_off_repeat' ELSE cls.removed_by END AS removed_by
        FROM ul JOIN cls USING (norm) LEFT JOIN cut USING (_row, norm)""")

    # per-form aggregates
    rule_cols = ",\n".join(f"count(*) FILTER (WHERE removed_by = '{k}') AS n_removed_{k}" for k in RULES)
    con.execute(f"""
        CREATE TABLE agg AS
        SELECT ulc._row,
               string_agg(CASE WHEN ulc.times_in_form > 1 AND r.finish_reason = 'stop'
                               THEN ulc.text || ' [×' || ulc.times_in_form || ']' ELSE ulc.text END,
                          chr(10) ORDER BY ulc.first_line_no) FILTER (WHERE ulc.removed_by IS NULL) AS remarks_clean,
               count(*) FILTER (WHERE ulc.removed_by IS NULL) AS n_lines_clean,
               {rule_cols}
        FROM ulc JOIN r USING (_row) GROUP BY ulc._row""")
    raw_counts = "SELECT _row, count(*) AS n_lines_raw FROM ln GROUP BY _row"
    zero = ", ".join(f"coalesce(a.n_removed_{k}, 0)::INTEGER AS n_removed_{k}" for k in RULES)
    con.execute(f"""
        COPY (
          SELECT r.* EXCLUDE (_row),
                 coalesce(a.remarks_clean, '') AS remarks_clean,
                 coalesce(a.n_lines_clean, 0) > 0 AS has_remarks_clean,
                 coalesce(c.n_lines_raw, 0)::INTEGER AS n_lines_raw,
                 coalesce(a.n_lines_clean, 0)::INTEGER AS n_lines_clean,
                 {zero}
          FROM r LEFT JOIN agg a USING (_row) LEFT JOIN ({raw_counts}) c USING (_row)
          ORDER BY r._row
        ) TO '{out / "ec_remarks_clean.parquet"}' (FORMAT parquet, COMPRESSION zstd)""")
    con.execute(f"""
        COPY (
          SELECT r.source_path, r.filename, r.climate_id, r.station, r.province_normalized, r.year, r.month, r.run,
                 r.alternate_scan, r.finish_reason, u.first_line_no AS line_no, u.text, u.times_in_form,
                 u.removed_by
          FROM ulc u JOIN r USING (_row)
          ORDER BY r._row, u.first_line_no
        ) TO '{out / "ec_remark_lines.parquet"}' (FORMAT parquet, COMPRESSION zstd)""")
    print("wrote", out / "ec_remarks_clean.parquet", "and", out / "ec_remark_lines.parquet")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
