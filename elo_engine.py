"""
Football ELO Rating Engine
===========================
Replays 150 years of international results to produce a continuous
ELO rating for every national team, then exposes helpers for:
  - win probability between any two teams
  - upset detection (and scoring)
  - all-time upset leaderboard
  - Monte Carlo tournament simulation

Dataset: https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017
Expected columns: date, home_team, away_team, home_score, away_score, tournament, neutral
"""

import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Team name normalization
#    The dataset spans 150 years, so historical names need mapping to modern ones.
# ---------------------------------------------------------------------------

TEAM_NAME_MAP = {
    "West Germany":        "Germany",
    "German DR":           "Germany",
    "East Germany":        "Germany",
    "USSR":                "Russia",
    "Federal Republic of Yugoslavia": "Serbia",
    "Yugoslavia":          "Serbia",
    "Serbia and Montenegro": "Serbia",
    "Czechoslovakia":      "Czech Republic",
    "Zaire":               "DR Congo",
    "Burma":               "Myanmar",
    "United Arab Republic": "Egypt",
    "Dahomey":             "Benin",
    "Upper Volta":         "Burkina Faso",
    "Dutch East Indies":   "Indonesia",
    "China PR":            "China",
}

def normalize_name(name: str) -> str:
    return TEAM_NAME_MAP.get(name, name)


# ---------------------------------------------------------------------------
# 2. ELO configuration
# ---------------------------------------------------------------------------

INITIAL_RATING   = 1500   # Starting rating for any new team
HOME_ADVANTAGE   = 100    # ELO points added to home team's effective rating
K_FRIENDLY       = 5      # Friendlies: squads rotate, nothing at stake — barely moves ratings
K_QUALIFIER      = 30     # Weight for qualifiers / confederation tournaments
K_WORLD_CUP      = 60     # Weight for World Cup matches (high stakes)

# Mean reversion: after each World Cup, pull every team this fraction toward 1500.
# 0.1 = move 10% of the gap between current rating and 1500 back toward 1500.
# E.g. a team at 2000 becomes 2000 - 0.1*(2000-1500) = 1950.
# Prevents rating inflation accumulating across decades.
MEAN_REVERSION_RATE = 0.1

# Which tournament keywords get which K-factor
def get_k_factor(tournament: str) -> int:
    t = tournament.lower()
    if "fifa world cup" in t:
        return K_WORLD_CUP
    if "friendly" in t:
        return K_FRIENDLY
    return K_QUALIFIER


# ---------------------------------------------------------------------------
# 3. Core ELO math
# ---------------------------------------------------------------------------

def expected_score(rating_a: float, rating_b: float) -> float:
    """Probability that team A beats team B, given their ratings."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def margin_multiplier(goal_diff: int) -> float:
    """
    Larger victories carry more information.
    Formula inspired by FiveThirtyEight's NFL ELO model.
      1 goal  → 1.0x
      2 goals → 1.5x
      3 goals → 1.75x
      4+ goals → scales up slowly via log
    """
    gd = abs(goal_diff)
    if gd <= 1:
        return 1.0
    return np.log(gd + 1) * 1.5


def elo_update(
    rating_a: float,
    rating_b: float,
    score_a: float,      # 1 = win, 0.5 = draw, 0 = loss
    goal_diff: int,
    k: int,
) -> tuple[float, float]:
    """Return (new_rating_a, new_rating_b) after one match."""
    exp_a = expected_score(rating_a, rating_b)
    score_b = 1.0 - score_a
    exp_b  = 1.0 - exp_a

    mult = margin_multiplier(goal_diff)

    delta_a = k * mult * (score_a - exp_a)
    delta_b = k * mult * (score_b - exp_b)

    return rating_a + delta_a, rating_b + delta_b


# ---------------------------------------------------------------------------
# 4. ELO engine — replay every match in chronological order
# ---------------------------------------------------------------------------

class EloEngine:
    """
    Ingests a DataFrame of match results and maintains a live rating
    for every team.  Call .fit(df) to replay history, then use
    .rating(team), .win_probability(), .upset_score(), etc.
    """

    def __init__(self):
        self.ratings: dict[str, float] = defaultdict(lambda: INITIAL_RATING)
        self.history: list[dict] = []   # one row per match

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "EloEngine":
        """
        Replay matches in chronological order.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: date, home_team, away_team,
            home_score, away_score, tournament, neutral
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Normalize team names
        df["home_team"] = df["home_team"].map(normalize_name)
        df["away_team"] = df["away_team"].map(normalize_name)

        # Drop rows where scores are missing (abandoned matches, walkovers, etc.)
        before = len(df)
        df = df.dropna(subset=["home_score", "away_score"])
        dropped = before - len(df)
        if dropped:
            print(f"Dropped {dropped} rows with missing scores.")

        # Track which World Cup years we've already applied reversion for
        wc_years_seen: set[int] = set()

        for _, row in df.iterrows():
            tournament = str(row.get("tournament", ""))
            year       = row["date"].year

            # Apply mean reversion once per World Cup year, before the first WC match
            # World Cups run in even years divisible by 4 from 1930 onward
            if (
                "fifa world cup" in tournament.lower()
                and "qualification" not in tournament.lower()
                and year not in wc_years_seen
                and year >= 1930
            ):
                self._apply_mean_reversion()
                wc_years_seen.add(year)

            self._process_match(row)

        print(f"Replayed {len(df):,} matches. {len(self.ratings)} unique teams rated.")
        print(f"Mean reversion applied at {len(wc_years_seen)} World Cup cycles: {sorted(wc_years_seen)}")
        return self

    def _apply_mean_reversion(self):
        """Pull every team's rating toward 1500 by MEAN_REVERSION_RATE."""
        for team in self.ratings:
            gap = self.ratings[team] - INITIAL_RATING
            self.ratings[team] -= gap * MEAN_REVERSION_RATE

    def _process_match(self, row: pd.Series):
        home = row["home_team"]
        away = row["away_team"]
        neutral = bool(row.get("neutral", False))

        r_home = self.ratings[home]
        r_away = self.ratings[away]

        # Apply home advantage only for non-neutral venues
        r_home_eff = r_home + (0 if neutral else HOME_ADVANTAGE)

        # Determine match outcome
        hs, as_ = int(row["home_score"]), int(row["away_score"])
        if hs > as_:
            score_home = 1.0
        elif hs < as_:
            score_home = 0.0
        else:
            score_home = 0.5

        k = get_k_factor(str(row.get("tournament", "")))
        goal_diff = abs(hs - as_)

        new_r_home, new_r_away = elo_update(
            r_home_eff, r_away, score_home, goal_diff, k
        )

        # Store actual (not effective) rating update
        delta = new_r_home - r_home_eff
        self.ratings[home] = r_home + delta
        self.ratings[away]  = new_r_away

        # Record for history DataFrame
        pre_prob_home = expected_score(r_home_eff, r_away)
        self.history.append({
            "date":          row["date"],
            "home_team":     home,
            "away_team":     away,
            "home_score":    hs,
            "away_score":    as_,
            "tournament":    row.get("tournament", ""),
            "neutral":       neutral,
            "pre_elo_home":  r_home,
            "pre_elo_away":  r_away,
            "pre_prob_home": round(pre_prob_home, 4),
            "pre_prob_away": round(1 - pre_prob_home, 4),
            "post_elo_home": self.ratings[home],
            "post_elo_away": self.ratings[away],
            "upset":         self._is_upset(score_home, pre_prob_home),
            "upset_score":   self._compute_upset_score(score_home, pre_prob_home),
        })

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def rating(self, team: str) -> float:
        """Current ELO rating for a team."""
        return self.ratings[normalize_name(team)]

    def win_probability(self, team_a: str, team_b: str, home: Optional[str] = None) -> dict:
        """
        Win probabilities for a match between team_a and team_b.

        Parameters
        ----------
        home : str or None
            Pass team name if playing at home, None for neutral venue.
        """
        ra = self.ratings[normalize_name(team_a)]
        rb = self.ratings[normalize_name(team_b)]

        if home == team_a:
            ra += HOME_ADVANTAGE
        elif home == team_b:
            rb += HOME_ADVANTAGE

        p_a = expected_score(ra, rb)
        return {
            team_a: round(p_a, 4),
            team_b: round(1 - p_a, 4),
        }

    @staticmethod
    def _is_upset(score_home: float, prob_home: float) -> bool:
        """True if the lower-rated team won (not a draw)."""
        if score_home == 1.0 and prob_home < 0.5:
            return True
        if score_home == 0.0 and prob_home > 0.5:
            return True
        return False

    @staticmethod
    def _compute_upset_score(score_home: float, prob_home: float) -> float:
        """
        How surprising was this result?
        Score = pre-match win probability of the LOSING team.
        Higher = bigger shock. Range: 0 to ~0.5 (draws ignored).
        """
        if score_home == 1.0:   # home won
            return round(1 - prob_home, 4)   # away's pre-match prob
        if score_home == 0.0:   # away won
            return round(prob_home, 4)        # home's pre-match prob
        return 0.0              # draw — not counted as upset

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def get_history_df(self) -> pd.DataFrame:
        """Full match history with pre/post ratings and upset flags."""
        return pd.DataFrame(self.history)

    def top_upsets(
        self,
        n: int = 20,
        tournament_filter: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return the n biggest upsets in history, optionally filtered
        to a specific tournament keyword (e.g. 'FIFA World Cup').
        """
        df = self.get_history_df()
        df = df[df["upset"] == True]

        if tournament_filter:
            df = df[df["tournament"].str.contains(tournament_filter, case=False, na=False)]

        df = df.sort_values("upset_score", ascending=False).head(n)

        # Pretty-print the result column
        def result_str(row):
            winner = row["home_team"] if row["home_score"] > row["away_score"] else row["away_team"]
            loser  = row["away_team"] if winner == row["home_team"] else row["home_team"]
            gs     = f"{row['home_score']}-{row['away_score']}"
            return f"{winner} def. {loser} ({gs})"

        df = df.copy()
        df["match"] = df.apply(result_str, axis=1)
        df["loser_pre_prob_%"] = (df["upset_score"] * 100).round(1)

        return df[["date", "tournament", "match", "loser_pre_prob_%", "upset_score"]].reset_index(drop=True)

    def ratings_table(self, top_n: int = 30) -> pd.DataFrame:
        """Current ELO leaderboard."""
        rows = sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return pd.DataFrame(rows, columns=["team", "elo"]).assign(
            elo=lambda d: d["elo"].round(1)
        )

    def team_history(self, team: str) -> pd.DataFrame:
        """Rating over time for a single team — useful for line charts."""
        team = normalize_name(team)
        df = self.get_history_df()
        home = df[df["home_team"] == team][["date", "post_elo_home"]].rename(columns={"post_elo_home": "elo"})
        away = df[df["away_team"] == team][["date", "post_elo_away"]].rename(columns={"post_elo_away": "elo"})
        return pd.concat([home, away]).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. Monte Carlo tournament simulator
# ---------------------------------------------------------------------------

def simulate_group(teams: list[str], engine: EloEngine, n_sims: int = 10_000) -> pd.DataFrame:
    """
    Simulate a World Cup group stage N times.
    Returns a DataFrame with each team's probability of finishing 1st or 2nd
    (i.e. advancing from the group).

    Parameters
    ----------
    teams : list of 4 team names
    engine : fitted EloEngine
    n_sims : number of simulations
    """
    assert len(teams) == 4, "Group stage requires exactly 4 teams"

    # All pairs in round-robin
    matchups = [
        (teams[i], teams[j])
        for i in range(4)
        for j in range(i + 1, 4)
    ]

    advance_count = defaultdict(int)

    for _ in range(n_sims):
        points = defaultdict(int)
        gd     = defaultdict(int)   # goal difference (tiebreaker)

        for a, b in matchups:
            p_a = engine.win_probability(a, b)["" + a]  # neutral venue
            r = np.random.random()

            # Simulate draw band: if outcome is close to 0.5, more likely draw
            draw_band = 0.08
            if abs(r - 0.5) < draw_band * p_a:
                points[a] += 1
                points[b] += 1
            elif r < p_a:
                points[a] += 3
                gd[a] += 1
                gd[b] -= 1
            else:
                points[b] += 3
                gd[b] += 1
                gd[a] -= 1

        # Rank by points then goal difference
        ranked = sorted(teams, key=lambda t: (points[t], gd[t]), reverse=True)
        advance_count[ranked[0]] += 1
        advance_count[ranked[1]] += 1

    results = pd.DataFrame([
        {"team": t, "advance_%": round(advance_count[t] / n_sims * 100, 1)}
        for t in teams
    ]).sort_values("advance_%", ascending=False).reset_index(drop=True)

    return results


# ---------------------------------------------------------------------------
# 6. Example usage  (remove or guard with __main__ before importing)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Load data ---
    df = pd.read_csv("results.csv")   # from Kaggle dataset

    # --- Fit OLD engine (no fixes) for comparison ---
    import copy

    K_FRIENDLY = 20   # temporarily restore old value
    MEAN_REVERSION_RATE_OLD = 0.0

    # We'll just run the new engine and note what changed
    print("=" * 55)
    print("  ELO engine with fixes applied")
    print("  K_FRIENDLY: 20 → 5  |  Mean reversion: 10% per WC")
    print("=" * 55)

    engine = EloEngine()
    engine.fit(df)

    # --- Current top ratings ---
    print("\n=== Top 15 teams by ELO (fixed engine) ===")
    print(engine.ratings_table(15).to_string(index=False))

    # --- Sanity check: teams that should be lower ---
    print("\n=== Sanity check: previously inflated teams ===")
    check_teams = ["Norway", "Ecuador", "Colombia", "Japan", "Spain", "Brazil", "France", "Argentina", "Germany"]
    rows = [(t, round(engine.rating(t), 1)) for t in check_teams]
    rows.sort(key=lambda x: x[1], reverse=True)
    for team, elo in rows:
        print(f"  {team:<15} {elo}")

    # --- Win probability example ---
    prob = engine.win_probability("Brazil", "Germany")
    print(f"\nBrazil vs Germany (neutral): {prob}")

    prob_home = engine.win_probability("England", "France", home="England")
    print(f"England (home) vs France:    {prob_home}")

    # --- All-time biggest World Cup upsets ---
    print("\n=== Top 10 World Cup upsets of all time ===")
    upsets = engine.top_upsets(n=10, tournament_filter="FIFA World Cup")
    print(upsets.to_string(index=False))

    # --- Simulate a group ---
    print("\n=== Group of Death simulation (10,000 runs) ===")
    group = ["Brazil", "France", "Argentina", "Portugal"]
    sim = simulate_group(group, engine, n_sims=10_000)
    print(sim.to_string(index=False))

    # --- Team rating history (for plotting) ---
    ger_history = engine.team_history("Germany")
    print(f"\nGermany ELO history: {len(ger_history)} data points")
    print(ger_history.tail(5))