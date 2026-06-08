"""
Football ELO Rating Engine
===========================
Replays 150 years of international results to produce a continuous
ELO rating for every national team, then exposes helpers for:
  - win probability between any two teams
  - upset detection (and scoring)
  - all-time upset leaderboard
  - Monte Carlo tournament simulation
  - classifier feature matrix (all-time ELO + 3-yr form + conf multiplier)

Dataset: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
Expected columns: date, home_team, away_team, home_score, away_score, tournament, neutral
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Team name normalization
# ---------------------------------------------------------------------------

TEAM_NAME_MAP = {
    "West Germany":                   "Germany",
    "German DR":                      "Germany",
    "East Germany":                   "Germany",
    "USSR":                           "Russia",
    "Federal Republic of Yugoslavia": "Serbia",
    "Yugoslavia":                     "Serbia",
    "Serbia and Montenegro":          "Serbia",
    "Czechoslovakia":                 "Czech Republic",
    "Zaire":                          "DR Congo",
    "Burma":                          "Myanmar",
    "United Arab Republic":           "Egypt",
    "Dahomey":                        "Benin",
    "Upper Volta":                    "Burkina Faso",
    "Dutch East Indies":              "Indonesia",
    "China PR":                       "China",
}

def normalize_name(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


# ---------------------------------------------------------------------------
# 2. Confederation lookup
#    Used to scale K-factor down when both teams are from a weaker pool.
#    Beating Bahrain in AFC qualifiers shouldn't be worth as much as
#    beating Sweden in UEFA qualifiers.
# ---------------------------------------------------------------------------

# Multiplier applied to K when BOTH teams are from that confederation.
# Cross-confederation matches (e.g. Japan vs Germany) use 1.0 — full weight.
CONF_K_MULTIPLIER = {
    "UEFA":     1.00,
    "CONMEBOL": 0.95,
    "CONCACAF": 0.80,
    "CAF":      0.75,
    "AFC":      0.70,
    "OFC":      0.60,
}

# fmt: off
TEAM_CONFEDERATION = {
    # UEFA
    "Albania": "UEFA", "Andorra": "UEFA", "Armenia": "UEFA", "Austria": "UEFA",
    "Azerbaijan": "UEFA", "Belarus": "UEFA", "Belgium": "UEFA", "Bosnia and Herzegovina": "UEFA",
    "Bulgaria": "UEFA", "Croatia": "UEFA", "Cyprus": "UEFA", "Czech Republic": "UEFA",
    "Denmark": "UEFA", "England": "UEFA", "Estonia": "UEFA", "Faroe Islands": "UEFA",
    "Finland": "UEFA", "France": "UEFA", "Georgia": "UEFA", "Germany": "UEFA",
    "Gibraltar": "UEFA", "Greece": "UEFA", "Hungary": "UEFA", "Iceland": "UEFA",
    "Ireland": "UEFA", "Israel": "UEFA", "Italy": "UEFA", "Kazakhstan": "UEFA",
    "Kosovo": "UEFA", "Latvia": "UEFA", "Liechtenstein": "UEFA", "Lithuania": "UEFA",
    "Luxembourg": "UEFA", "Malta": "UEFA", "Moldova": "UEFA", "Montenegro": "UEFA",
    "Netherlands": "UEFA", "North Macedonia": "UEFA", "Northern Ireland": "UEFA",
    "Norway": "UEFA", "Poland": "UEFA", "Portugal": "UEFA", "Romania": "UEFA",
    "Russia": "UEFA", "San Marino": "UEFA", "Scotland": "UEFA", "Serbia": "UEFA",
    "Slovakia": "UEFA", "Slovenia": "UEFA", "Spain": "UEFA", "Sweden": "UEFA",
    "Switzerland": "UEFA", "Turkey": "UEFA", "Ukraine": "UEFA", "Wales": "UEFA",
    # CONMEBOL
    "Argentina": "CONMEBOL", "Bolivia": "CONMEBOL", "Brazil": "CONMEBOL",
    "Chile": "CONMEBOL", "Colombia": "CONMEBOL", "Ecuador": "CONMEBOL",
    "Paraguay": "CONMEBOL", "Peru": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Venezuela": "CONMEBOL",
    # CONCACAF
    "Canada": "CONCACAF", "Costa Rica": "CONCACAF", "Cuba": "CONCACAF",
    "El Salvador": "CONCACAF", "Guatemala": "CONCACAF", "Haiti": "CONCACAF",
    "Honduras": "CONCACAF", "Jamaica": "CONCACAF", "Mexico": "CONCACAF",
    "Panama": "CONCACAF", "Trinidad and Tobago": "CONCACAF",
    "United States": "CONCACAF",
    # CAF
    "Algeria": "CAF", "Angola": "CAF", "Benin": "CAF", "Burkina Faso": "CAF",
    "Cameroon": "CAF", "Cape Verde": "CAF", "Central African Republic": "CAF",
    "DR Congo": "CAF", "Egypt": "CAF", "Ethiopia": "CAF", "Gabon": "CAF",
    "Ghana": "CAF", "Guinea": "CAF", "Ivory Coast": "CAF", "Kenya": "CAF",
    "Libya": "CAF", "Mali": "CAF", "Morocco": "CAF", "Mozambique": "CAF",
    "Nigeria": "CAF", "Senegal": "CAF", "Sierra Leone": "CAF", "Somalia": "CAF",
    "South Africa": "CAF", "Tanzania": "CAF", "Togo": "CAF", "Tunisia": "CAF",
    "Uganda": "CAF", "Zambia": "CAF", "Zimbabwe": "CAF",
    # AFC
    "Australia": "AFC", "Bahrain": "AFC", "China": "AFC", "India": "AFC",
    "Indonesia": "AFC", "Iran": "AFC", "Iraq": "AFC", "Japan": "AFC",
    "Jordan": "AFC", "Kuwait": "AFC", "Lebanon": "AFC", "Malaysia": "AFC",
    "Myanmar": "AFC", "North Korea": "AFC", "Oman": "AFC", "Pakistan": "AFC",
    "Palestine": "AFC", "Philippines": "AFC", "Qatar": "AFC", "Saudi Arabia": "AFC",
    "Singapore": "AFC", "South Korea": "AFC", "Syria": "AFC", "Thailand": "AFC",
    "United Arab Emirates": "AFC", "Uzbekistan": "AFC", "Vietnam": "AFC",
    "Yemen": "AFC",
    # OFC
    "Fiji": "OFC", "New Caledonia": "OFC", "New Zealand": "OFC",
    "Papua New Guinea": "OFC", "Solomon Islands": "OFC", "Vanuatu": "OFC",
}
# fmt: on

def get_confederation(team: str) -> Optional[str]:
    return TEAM_CONFEDERATION.get(normalize_name(team))

def confederation_k_multiplier(home: str, away: str) -> float:
    """
    Returns a K multiplier based on the weaker confederation.
    If both teams are from the same (weaker) confederation, apply that
    confederation's discount — their match pool is less competitive.
    Cross-confederation matches always use 1.0 (full weight).
    """
    conf_h = get_confederation(home)
    conf_a = get_confederation(away)
    if conf_h and conf_a and conf_h == conf_a:
        return CONF_K_MULTIPLIER.get(conf_h, 1.0)
    return 1.0   # cross-confederation: full weight


# ---------------------------------------------------------------------------
# 3. ELO configuration
# ---------------------------------------------------------------------------

INITIAL_RATING      = 1500
HOME_ADVANTAGE      = 100
K_FRIENDLY          = 5      # near-zero: friendlies are noise
K_QUALIFIER         = 30
K_WORLD_CUP         = 60
MEAN_REVERSION_RATE = 0.1    # 10% pull toward 1500 per World Cup cycle
FORM_WINDOW_DAYS    = 365 * 3  # 3-year rolling window for form ELO feature


def get_k_factor(tournament: str) -> float:
    t = tournament.lower()
    if "fifa world cup" in t and "qualif" not in t:
        return K_WORLD_CUP
    if "friendly" in t:
        return K_FRIENDLY
    return K_QUALIFIER


# ---------------------------------------------------------------------------
# 4. Core ELO math
# ---------------------------------------------------------------------------

def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def margin_multiplier(goal_diff: int) -> float:
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    return np.log(gd + 1) * 1.5


def elo_update(
    rating_a: float,
    rating_b: float,
    score_a: float,
    goal_diff: int,
    k: float,
) -> tuple[float, float]:
    exp_a  = expected_score(rating_a, rating_b)
    mult   = margin_multiplier(goal_diff)
    delta_a = k * mult * (score_a - exp_a)
    delta_b = k * mult * ((1 - score_a) - (1 - exp_a))
    return rating_a + delta_a, rating_b + delta_b


# ---------------------------------------------------------------------------
# 5. ELO engine
# ---------------------------------------------------------------------------

class EloEngine:
    """
    Full ELO engine with:
      - confederation K-factor multiplier
      - mean reversion at each World Cup cycle
      - 3-year rolling form rating (stored per-match for feature engineering)
    """

    def __init__(self):
        self.ratings: dict[str, float] = defaultdict(lambda: INITIAL_RATING)
        # form_ratings tracks a separate ELO that only uses the last 3 years
        # of matches. Recomputed in build_classifier_features(), not live.
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "EloEngine":
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df["home_team"] = df["home_team"].map(normalize_name)
        df["away_team"] = df["away_team"].map(normalize_name)

        before = len(df)
        df = df.dropna(subset=["home_score", "away_score"])
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped} rows with missing scores.")

        wc_years_seen: set[int] = set()

        for _, row in df.iterrows():
            tournament = str(row.get("tournament", ""))
            year       = row["date"].year

            if (
                "fifa world cup" in tournament.lower()
                and "qualif" not in tournament.lower()
                and year not in wc_years_seen
                and year >= 1930
            ):
                self._apply_mean_reversion()
                wc_years_seen.add(year)

            self._process_match(row)

        print(f"Replayed {len(df):,} matches. {len(self.ratings)} unique teams rated.")
        print(f"Mean reversion applied at {len(wc_years_seen)} WC cycles: {sorted(wc_years_seen)}")
        return self

    def _apply_mean_reversion(self):
        for team in self.ratings:
            gap = self.ratings[team] - INITIAL_RATING
            self.ratings[team] -= gap * MEAN_REVERSION_RATE

    def _process_match(self, row: pd.Series):
        home    = row["home_team"]
        away    = row["away_team"]
        neutral = bool(row.get("neutral", False))

        r_home = self.ratings[home]
        r_away = self.ratings[away]
        r_home_eff = r_home + (0 if neutral else HOME_ADVANTAGE)

        hs, as_ = int(row["home_score"]), int(row["away_score"])
        score_home = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        goal_diff  = abs(hs - as_)

        # K scaled by tournament tier AND confederation strength
        base_k  = get_k_factor(str(row.get("tournament", "")))
        conf_mult = confederation_k_multiplier(home, away)
        k = base_k * conf_mult

        new_r_home_eff, new_r_away = elo_update(r_home_eff, r_away, score_home, goal_diff, k)

        delta = new_r_home_eff - r_home_eff
        self.ratings[home] = r_home + delta
        self.ratings[away] = new_r_away

        pre_prob_home = expected_score(r_home_eff, r_away)
        self.history.append({
            "date":           row["date"],
            "home_team":      home,
            "away_team":      away,
            "home_score":     hs,
            "away_score":     as_,
            "tournament":     row.get("tournament", ""),
            "neutral":        neutral,
            "conf_home":      get_confederation(home),
            "conf_away":      get_confederation(away),
            "conf_k_mult":    round(conf_mult, 3),
            "effective_k":    round(k, 1),
            "pre_elo_home":   r_home,
            "pre_elo_away":   r_away,
            "pre_prob_home":  round(pre_prob_home, 4),
            "pre_prob_away":  round(1 - pre_prob_home, 4),
            "post_elo_home":  self.ratings[home],
            "post_elo_away":  self.ratings[away],
            "upset":          self._is_upset(score_home, pre_prob_home),
            "upset_score":    self._compute_upset_score(score_home, pre_prob_home),
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def rating(self, team: str) -> float:
        return self.ratings[normalize_name(team)]

    def win_probability(self, team_a: str, team_b: str, home: Optional[str] = None) -> dict:
        ra = self.ratings[normalize_name(team_a)]
        rb = self.ratings[normalize_name(team_b)]
        if home == team_a:
            ra += HOME_ADVANTAGE
        elif home == team_b:
            rb += HOME_ADVANTAGE
        p_a = expected_score(ra, rb)
        return {team_a: round(p_a, 4), team_b: round(1 - p_a, 4)}

    @staticmethod
    def _is_upset(score_home: float, prob_home: float) -> bool:
        if score_home == 1.0 and prob_home < 0.5:
            return True
        if score_home == 0.0 and prob_home > 0.5:
            return True
        return False

    @staticmethod
    def _compute_upset_score(score_home: float, prob_home: float) -> float:
        if score_home == 1.0:
            return round(1 - prob_home, 4)
        if score_home == 0.0:
            return round(prob_home, 4)
        return 0.0

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def ratings_table(self, top_n: int = 30) -> pd.DataFrame:
        rows = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return pd.DataFrame(rows, columns=["team", "elo"]).assign(
            elo=lambda d: d["elo"].round(1)
        )

    def team_history(self, team: str) -> pd.DataFrame:
        team = normalize_name(team)
        df   = self.get_history_df()
        home = df[df["home_team"] == team][["date", "post_elo_home"]].rename(columns={"post_elo_home": "elo"})
        away = df[df["away_team"] == team][["date", "post_elo_away"]].rename(columns={"post_elo_away": "elo"})
        return pd.concat([home, away]).sort_values("date").reset_index(drop=True)

    def top_upsets(self, n: int = 20, tournament_filter: Optional[str] = None) -> pd.DataFrame:
        df = self.get_history_df()
        df = df[df["upset"] == True]
        if tournament_filter:
            df = df[df["tournament"].str.contains(tournament_filter, case=False, na=False)]
        df = df.sort_values("upset_score", ascending=False).head(n).copy()

        def result_str(row):
            winner = row["home_team"] if row["home_score"] > row["away_score"] else row["away_team"]
            loser  = row["away_team"] if winner == row["home_team"] else row["home_team"]
            return f"{winner} def. {loser} ({row['home_score']}-{row['away_score']})"

        df["match"]            = df.apply(result_str, axis=1)
        df["loser_pre_prob_%"] = (df["upset_score"] * 100).round(1)
        return df[["date", "tournament", "match", "loser_pre_prob_%", "upset_score"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Classifier feature builder
#    Produces one row per match with all features needed for the upset model.
#    Key features:
#      elo_diff          — all-time ELO gap (home minus away, pre-match)
#      form_elo_diff     — 3-year rolling ELO gap (recency-weighted)
#      both_from_conf    — 1 if same confederation (weaker pool flag)
#      conf_k_mult       — the actual multiplier used (continuous version)
#      is_neutral        — venue effect removed
#      tournament_weight — K-tier of the match (proxy for stakes)
#      target            — 1 if upset occurred, 0 otherwise
# ---------------------------------------------------------------------------

def build_classifier_features(engine: EloEngine) -> pd.DataFrame:
    """
    Returns a feature matrix ready for scikit-learn / XGBoost.

    The 3-year form ELO is computed by re-running a lightweight ELO
    on only the matches within a 3-year window before each match date.
    This is O(n²) in the worst case but fast in practice because the
    window is bounded — typically ~3,000–5,000 matches per window.

    For the classifier:
      - Use elo_diff + form_elo_diff as separate features (let the model
        decide how to weight recent vs all-time form).
      - Use conf_k_mult as a feature so the model learns that
        same-confederation upsets in weak pools are less surprising.
    """
    history = engine.get_history_df()
    history = history.sort_values("date").reset_index(drop=True)

    # --- Compute 3-year form ELO for every match ---
    # We replay a mini ELO engine using only matches within the window.
    # This gives each team a "current form" rating at the time of each match.
    print("Computing 3-year form ELO (this takes ~30–60 seconds)...")

    window = pd.Timedelta(days=FORM_WINDOW_DAYS)
    dates  = history["date"].values

    # Build a fast index: for each match i, find all matches in [date-3yr, date)
    form_elo_home_list = []
    form_elo_away_list = []

    for i, row in history.iterrows():
        cutoff_start = row["date"] - window
        cutoff_end   = row["date"]

        mask = (history["date"] >= cutoff_start) & (history["date"] < cutoff_end)
        recent = history[mask]

        # Mini ELO replay on the window
        mini_ratings: dict[str, float] = defaultdict(lambda: INITIAL_RATING)
        for _, m in recent.iterrows():
            rh = mini_ratings[m["home_team"]]
            ra = mini_ratings[m["away_team"]]
            hs_, as__ = m["home_score"], m["away_score"]
            sc = 1.0 if hs_ > as__ else (0.0 if hs_ < as__ else 0.5)
            k  = get_k_factor(str(m["tournament"])) * m["conf_k_mult"]
            new_rh, new_ra = elo_update(rh, ra, sc, abs(int(hs_) - int(as__)), k)
            mini_ratings[m["home_team"]] = new_rh
            mini_ratings[m["away_team"]] = new_ra

        form_elo_home_list.append(mini_ratings[row["home_team"]])
        form_elo_away_list.append(mini_ratings[row["away_team"]])

        if i % 5000 == 0 and i > 0:
            print(f"  {i:,} / {len(history):,} matches processed...")

    history["form_elo_home"] = form_elo_home_list
    history["form_elo_away"] = form_elo_away_list

    # --- Assemble feature columns ---
    features = pd.DataFrame({
        "date":             history["date"],
        "home_team":        history["home_team"],
        "away_team":        history["away_team"],
        "tournament":       history["tournament"],

        # Core ELO features
        "elo_diff":         (history["pre_elo_home"] - history["pre_elo_away"]).round(1),
        "elo_home":         history["pre_elo_home"].round(1),
        "elo_away":         history["pre_elo_away"].round(1),

        # 3-year form features
        "form_elo_diff":    (history["form_elo_home"] - history["form_elo_away"]).round(1),
        "form_elo_home":    history["form_elo_home"].round(1),
        "form_elo_away":    history["form_elo_away"].round(1),

        # Confederation context
        "conf_home":        history["conf_home"],
        "conf_away":        history["conf_away"],
        "same_conf":        (history["conf_home"] == history["conf_away"]).astype(int),
        "conf_k_mult":      history["conf_k_mult"],

        # Match context
        "is_neutral":       history["neutral"].astype(int),
        "tournament_k":     history["effective_k"],   # stake proxy

        # Win probability (derived from all-time ELO — useful as direct feature)
        "pre_prob_home":    history["pre_prob_home"],
        "pre_prob_away":    history["pre_prob_away"],

        # Target
        "target":           history["upset"].astype(int),
        "upset_score":      history["upset_score"],   # continuous alternative target
    })

    print(f"Feature matrix ready: {len(features):,} rows × {len(features.columns)} columns.")
    print(f"Upset rate in dataset: {features['target'].mean():.1%}")
    return features


# ---------------------------------------------------------------------------
# 7. Monte Carlo tournament simulator
# ---------------------------------------------------------------------------

def simulate_group(teams: list[str], engine: EloEngine, n_sims: int = 10_000) -> pd.DataFrame:
    assert len(teams) == 4, "Group stage requires exactly 4 teams"
    matchups = [(teams[i], teams[j]) for i in range(4) for j in range(i + 1, 4)]
    advance_count: dict[str, int] = defaultdict(int)

    for _ in range(n_sims):
        points: dict[str, int] = defaultdict(int)
        gd:     dict[str, int] = defaultdict(int)

        for a, b in matchups:
            p_a = engine.win_probability(a, b)[a]
            r   = np.random.random()
            draw_band = 0.08
            if abs(r - 0.5) < draw_band * p_a:
                points[a] += 1; points[b] += 1
            elif r < p_a:
                points[a] += 3; gd[a] += 1; gd[b] -= 1
            else:
                points[b] += 3; gd[b] += 1; gd[a] -= 1

        ranked = sorted(teams, key=lambda t: (points[t], gd[t]), reverse=True)
        advance_count[ranked[0]] += 1
        advance_count[ranked[1]] += 1

    return pd.DataFrame([
        {"team": t, "advance_%": round(advance_count[t] / n_sims * 100, 1)}
        for t in teams
    ]).sort_values("advance_%", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 8. Example usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = pd.read_csv("results.csv")

    print("=" * 60)
    print("  ELO engine v3: conf multiplier + mean reversion + form ELO")
    print("=" * 60)

    engine = EloEngine()
    engine.fit(df)

    print("\n=== Top 15 teams by all-time ELO ===")
    print(engine.ratings_table(15).to_string(index=False))

    print("\n=== Sanity check ===")
    check = ["Spain", "France", "Argentina", "Brazil", "Germany",
             "England", "Japan", "Colombia", "Ecuador", "Norway"]
    rows = sorted([(t, round(engine.rating(t), 1)) for t in check],
                  key=lambda x: x[1], reverse=True)
    for team, elo in rows:
        conf = get_confederation(team) or "unknown"
        print(f"  {team:<20} {elo:>7}  [{conf}]")

    print("\n=== Top 10 World Cup upsets ===")
    print(engine.top_upsets(10, "FIFA World Cup").to_string(index=False))

    print("\n=== Group sim: Brazil / France / Argentina / Portugal ===")
    print(simulate_group(["Brazil", "France", "Argentina", "Portugal"], engine).to_string(index=False))

    # Build classifier features (slow — comment out if just testing the engine)
    print("\n=== Building classifier feature matrix ===")
    features = build_classifier_features(engine)
    print(features[["date", "home_team", "away_team", "elo_diff",
                     "form_elo_diff", "conf_k_mult", "same_conf",
                     "pre_prob_home", "target"]].tail(10).to_string(index=False))

    # Save for use in the classifier notebook
    features.to_csv("classifier_features.csv", index=False)
    print("\nSaved → classifier_features.csv")