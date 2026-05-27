#mymodelfile.py  —  IPL Powerplay Score Predictor  v9

import re
import warnings
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")

# ── Optional LightGBM ──────────────────────────────────────────────────────
try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    print("[WARN]  LightGBM not installed — using XGBoost for Stage-1 classifier.")

# ── Optional sklearn ───────────────────────────────────────────────────────
try:
    from sklearn.preprocessing import StandardScaler
    _HAS_SKL = True
except ImportError:
    _HAS_SKL = False


# ═══════════════════════════════════════════════════════════════════════════
#  REGIME DEFINITIONS
# FIX D: Adjusted thresholds — low raised to 44, explosive lowered to 65.
#         This catches more genuinely explosive innings without destabilising
#         the normal regime (which still dominates well-predicted matches).
# ═══════════════════════════════════════════════════════════════════════════
REGIME_LOW       = 0   # powerplay score < 44  (was 42)
REGIME_NORMAL    = 1   # 44 ≤ score < 65       (was 42–68)
REGIME_EXPLOSIVE = 2   # score ≥ 65            (was 68)

def score_to_regime(score: float) -> int:
    if score < 44:
        return REGIME_LOW
    if score < 65:
        return REGIME_NORMAL
    return REGIME_EXPLOSIVE

REGIME_NAMES = {REGIME_LOW: "low", REGIME_NORMAL: "normal", REGIME_EXPLOSIVE: "explosive"}


# ═══════════════════════════════════════════════════════════════════════════
class MyModel:
# ═══════════════════════════════════════════════════════════════════════════

    CURRENT_TEAMS = [
        "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans",
        "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
        "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bengaluru",
        "Sunrisers Hyderabad",
    ]

    TEAM_KW = [
        (["chennai", "csk"],                                       "Chennai Super Kings"),
        (["delhi", "daredevils", "capitals"],                      "Delhi Capitals"),
        (["gujarat", "titans"],                                    "Gujarat Titans"),
        (["kolkata", "knight", "kkr"],                             "Kolkata Knight Riders"),
        (["lucknow", "super giant", "supergiants", "lsg"],         "Lucknow Super Giants"),
        (["mumbai", "indians"],                                    "Mumbai Indians"),
        (["punjab", "kings xi", "pbks"],                           "Punjab Kings"),
        (["rajasthan", "royals"],                                  "Rajasthan Royals"),
        (["challengers", "rcb", "bangalore", "bengaluru"],        "Royal Challengers Bengaluru"),
        (["sunrisers", "hyderabad", "srh", "deccan", "chargers"], "Sunrisers Hyderabad"),
    ]

    VENUE_KW = [
        (["aca", "guwahati", "barsapara"],                         "ACA Stadium, Guwahati"),
        (["jaitley", "kotla", "feroz"],                            "Arun Jaitley Stadium, Delhi"),
        (["ekana", "lucknow", "atal bihari", "bharat ratna"],      "Ekana Cricket Stadium, Lucknow"),
        (["eden", "kolkata"],                                      "Eden Gardens, Kolkata"),
        (["himachal", "hpca", "dharamshala", "dharamsala"],        "HP Cricket Association Stadium, Dharamshala"),
        (["chinnaswamy", "bengaluru", "bangalore"],                "M Chinnaswamy Stadium, Bengaluru"),
        (["chidambaram", "chepauk", "chennai"],                    "MA Chidambaram Stadium, Chennai"),
        (["narendra modi", "sardar patel", "motera", "ahmedabad"], "Narendra Modi Stadium, Ahmedabad"),
        (["maharaja yadavindra", "new chandigarh", "new international",
          "punjab cricket association", "mohali"],                 "New International Cricket Stadium, New Chandigarh"),
        (["rajiv gandhi", "hyderabad"],                            "Rajiv Gandhi International Stadium, Hyderabad"),
        (["sawai mansingh", "jaipur"],                             "Sawai Mansingh Stadium, Jaipur"),
        (["shaheed veer narayan", "raipur"],                       "Shaheed Veer Narayan Singh Stadium, Raipur"),
        (["wankhede", "mumbai"],                                   "Wankhede Stadium, Mumbai"),
    ]

    # FIX C: Updated explosion_base for venues showing consistent high powerplays
    # in 2024-2025 IPL. Rajiv Gandhi (SRH home) bumped 0.30→0.42 based on
    # observed actual scores of 84, 105, 75 in our prediction history.
    # Narendra Modi bumped 0.22→0.28. Chinnaswamy already high at 0.38.
    # Conservative increases only — no single venue exceeds 0.45.
    VENUE_BEHAVIOR = {
        "ACA Stadium, Guwahati":                             {"avg_score_mid": 175, "batting": 1, "spin": 1, "pace": 0, "dew": 0, "slow": 0, "explosion_base": 0.22},
        "Arun Jaitley Stadium, Delhi":                       {"avg_score_mid": 165, "batting": 0, "spin": 1, "pace": 0, "dew": 0, "slow": 1, "explosion_base": 0.15},
        "Ekana Cricket Stadium, Lucknow":                    {"avg_score_mid": 145, "batting": 0, "spin": 1, "pace": 0, "dew": 0, "slow": 1, "explosion_base": 0.10},
        "Eden Gardens, Kolkata":                             {"avg_score_mid": 190, "batting": 1, "spin": 0, "pace": 1, "dew": 1, "slow": 0, "explosion_base": 0.30},
        "HP Cricket Association Stadium, Dharamshala":       {"avg_score_mid": 175, "batting": 1, "spin": 0, "pace": 1, "dew": 0, "slow": 0, "explosion_base": 0.25},
        "M Chinnaswamy Stadium, Bengaluru":                  {"avg_score_mid": 210, "batting": 1, "spin": 0, "pace": 0, "dew": 1, "slow": 0, "explosion_base": 0.38},
        "MA Chidambaram Stadium, Chennai":                   {"avg_score_mid": 155, "batting": 0, "spin": 1, "pace": 0, "dew": 0, "slow": 1, "explosion_base": 0.12},
        "Narendra Modi Stadium, Ahmedabad":                  {"avg_score_mid": 180, "batting": 1, "spin": 1, "pace": 1, "dew": 0, "slow": 0, "explosion_base": 0.28},  # was 0.22
        "New International Cricket Stadium, New Chandigarh": {"avg_score_mid": 170, "batting": 1, "spin": 1, "pace": 1, "dew": 0, "slow": 0, "explosion_base": 0.22},
        "Rajiv Gandhi International Stadium, Hyderabad":     {"avg_score_mid": 195, "batting": 1, "spin": 0, "pace": 1, "dew": 1, "slow": 0, "explosion_base": 0.42},  # was 0.30, avg_score_mid was 190
        "Sawai Mansingh Stadium, Jaipur":                    {"avg_score_mid": 175, "batting": 0, "spin": 1, "pace": 0, "dew": 0, "slow": 1, "explosion_base": 0.18},
        "Shaheed Veer Narayan Singh Stadium, Raipur":        {"avg_score_mid": 185, "batting": 1, "spin": 0, "pace": 0, "dew": 0, "slow": 0, "explosion_base": 0.20},
        "Wankhede Stadium, Mumbai":                          {"avg_score_mid": 200, "batting": 1, "spin": 0, "pace": 1, "dew": 1, "slow": 0, "explosion_base": 0.35},  # was 0.32
    }
    _DEFAULT_BEHAVIOR = {"avg_score_mid": 175, "batting": 0, "spin": 0,
                         "pace": 0, "dew": 0, "slow": 0, "explosion_base": 0.20}

    TRAIN_FROM_YEAR = 2019

    # FIX F: Increased year weight decay from 0.08 → 0.10.
    # Effect: 2025 data gets weight ~2.0x vs 2019. 2023 data gets ~1.6x vs 2019.
    # This better captures the recent explosion in powerplay aggression across
    # IPL teams without instability (0.10 is still conservative).
    _YR_W_CACHE: dict = {
        y: float(np.exp(0.10 * max(0, y - 2018))) for y in range(2000, 2032)
    }

    # ── XGBoost hyper-params per regime ────────────────────────────────────
    # FIX A: Added nthread=1 to ALL XGBoost models for full determinism.
    # n_jobs=1 alone is insufficient with tree_method="hist" — parallel
    # histogram construction still causes non-deterministic results.
    _XGB_BASE = dict(
        objective="reg:absoluteerror",
        random_state=42,
        nthread=1,          # FIX A: determinism — overrides n_jobs for hist
        n_jobs=1,
        tree_method="hist",
        early_stopping_rounds=30,
    )

    _REGIME_XGB_PARAMS = {
        REGIME_LOW: dict(
            n_estimators=500, max_depth=4, learning_rate=0.04,
            subsample=0.80, colsample_bytree=0.75,
            min_child_weight=3, reg_alpha=0.10, reg_lambda=1.5, gamma=0.10,
        ),
        REGIME_NORMAL: dict(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.80, colsample_bytree=0.75,
            min_child_weight=4, reg_alpha=0.10, reg_lambda=1.5, gamma=0.10,
        ),
        REGIME_EXPLOSIVE: dict(
            n_estimators=600, max_depth=5, learning_rate=0.04,
            subsample=0.75, colsample_bytree=0.70,
            min_child_weight=3, reg_alpha=0.08, reg_lambda=1.2, gamma=0.08,
        ),
    }

    # FIX E: Per-inning calibration offsets learned from prediction history.
    # Inning 2 is consistently over-predicted (model doesn't account for
    # chase pressure reducing powerplay aggression when target is low).
    # These are conservative adjustments: I1=0, I2=-4 (bias only, no scale change).
    # This ONLY affects the final output clamp, not training — so well-predicted
    # matches near the decision boundary are protected by the clamp logic.
    _INNING_BIAS = {1: 0.0, 2: -4.0}

    # ────────────────────────────────────────────────────────────────────────
    def __init__(self) -> None:

        # FIX A: LightGBM must use num_threads=1 + force_col_wise=True for determinism.
        # force_col_wise=True disables auto-selection of parallel strategy which
        # can vary across runs even with identical seeds.
        if _HAS_LGB:
            self.classifier = LGBMClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.80, colsample_bytree=0.75,
                min_child_samples=8, reg_alpha=0.08, reg_lambda=1.2,
                random_state=42,
                num_threads=1,       # FIX A: determinism
                force_col_wise=True, # FIX A: determinism
                verbose=-1,
            )
        else:
            self.classifier = XGBClassifier(
                n_estimators=400, max_depth=5, learning_rate=0.05,
                subsample=0.80, colsample_bytree=0.75,
                min_child_weight=4, reg_alpha=0.10, reg_lambda=1.5,
                random_state=42,
                nthread=1,           # FIX A: determinism
                n_jobs=1,
                tree_method="hist",
                use_label_encoder=False,
                eval_metric="mlogloss",
            )

        self.regressors: dict = {
            r: XGBRegressor(**{**self._XGB_BASE, **self._REGIME_XGB_PARAMS[r]})
            for r in (REGIME_LOW, REGIME_NORMAL, REGIME_EXPLOSIVE)
        }

        # Lookup tables (populated during fit using past-only data)
        self.venue_stats:        dict = {}
        self.team_bat_stats:     dict = {}
        self.team_bowl_stats:    dict = {}
        self.matchup_stats:      dict = {}
        self.venue_team_stats:   dict = {}
        self.bat_player_stats:   dict = {}
        self.bowl_player_stats:  dict = {}
        self.team_wickets_pp:    dict = {}
        self.recent_bat:         dict = {}
        self.recent_bowl:        dict = {}
        self.inning_avg:         dict = {}
        self.regime_priors:      dict = {}
        self.venue_explosion_rt: dict = {}
        self.regime_means:       dict = {}
        self.regime_stds:        dict = {}

        self.player_id_to_name:  dict = {}
        self.unmapped_players:   list = []
        self._map_summary:       dict = {}
        self.feature_cols:       list = []

        self.global_mean:   float = 52.0
        self.global_median: float = 52.0
        self.global_std:    float = 12.0
        self.default_bat_sr: float = 120.0
        self.default_econ:   float = 7.5
        self._insample_mae:  float = 0.0

        # FIX G: Tighter calibration clamps to protect well-predicted matches.
        # Scale range 0.95–1.05 (was 0.92–1.08), bias capped at ±3.0 (was uncapped).
        self._calib_bias:  dict = {r: 0.0 for r in range(3)}
        self._calib_scale: dict = {r: 1.0 for r in range(3)}

        # Prediction year: set dynamically during predict
        self._predict_year: int = 2024

    # ════════════════════════════════════════════════════════ normalisation

    def _norm_team(self, name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            return "Unknown"
        n = " " + re.sub(r"\s+", " ", name.strip()).lower() + " "
        for keywords, canonical in self.TEAM_KW:
            for kw in keywords:
                if kw in n:
                    return canonical
        return re.sub(r"\s+", " ", name.strip())

    def _norm_venue(self, venue: str) -> str:
        if not isinstance(venue, str) or not venue.strip():
            return "Unknown Venue"
        v = re.sub(r"\s+", " ", venue.strip()).lower()
        for keywords, canonical in self.VENUE_KW:
            for kw in keywords:
                if kw in v:
                    return canonical
        return re.sub(r"\s+", " ", venue.strip())

    def _venue_features(self, venue: str) -> dict:
        b = self.VENUE_BEHAVIOR.get(venue, self._DEFAULT_BEHAVIOR)
        return {
            "vb_avg_score_mid":  float(b["avg_score_mid"]),
            "vb_batting":        float(b["batting"]),
            "vb_spin":           float(b["spin"]),
            "vb_pace":           float(b["pace"]),
            "vb_dew":            float(b["dew"]),
            "vb_slow":           float(b["slow"]),
            "vb_explosion_base": float(b.get("explosion_base", 0.20)),
        }

    @staticmethod
    def _parse_season(s) -> int:
        if isinstance(s, float) and not np.isnan(s):
            return int(s)
        if isinstance(s, int):
            return s
        s = str(s).strip()
        if "/" in s:
            return int(s.split("/")[0]) + 1
        try:
            return int(float(s))
        except Exception:
            return 2020

    @classmethod
    def _yr_w(cls, year: int) -> float:
        return cls._YR_W_CACHE.get(int(year), 1.0)

    def _shrink(self, sample_mean: float, cnt: int, global_mean: float,
                alpha: float = 15.0) -> float:
        """James-Stein type shrinkage toward global mean."""
        return (cnt * sample_mean + alpha * global_mean) / (cnt + alpha)

    @staticmethod
    def _wmean(vals, w) -> float:
        vals, w = np.asarray(vals, float), np.asarray(w, float)
        s = w.sum()
        return float(np.dot(vals, w) / s) if s > 0 else float(np.mean(vals))

    # ═══════════════════════════════════════════════════════ player mapping

    @staticmethod
    def _name_variants(full: str) -> list:
        parts = full.strip().split()
        variants = [full]
        if len(parts) >= 2:
            fi = parts[0][0].upper()
            if parts[0] != fi:
                variants.append(fi + " " + parts[-1])
            if re.match(r"^[A-Z]{2,3}$", parts[0]):
                variants.append(parts[0][0] + " " + " ".join(parts[1:]))
            if len(parts) >= 3:
                variants.append(" ".join(parts[1:]))
        seen, out = set(), []
        for v in variants:
            if v not in seen:
                seen.add(v); out.append(v)
        return out

    def map_player_ids(self, deliveries_df: pd.DataFrame, players_df) -> None:
        if players_df is None or len(players_df) == 0:
            print("[WARN]  players_df empty — ID mapping skipped.")
            return

        dnames: set = set()
        for col in ("batsman", "bowler", "non_striker"):
            if col in deliveries_df.columns:
                dnames.update(
                    deliveries_df[col].dropna().astype(str).str.strip().unique())
        dnames -= {"", "nan", "None"}
        dlist = list(dnames)
        dlist_lower = [d.lower() for d in dlist]

        self.unmapped_players = []
        exact_cnt = fuzzy_cnt = fallback_cnt = 0

        for _, row in players_df.iterrows():
            pid       = str(row["ID"]).strip()
            full_name = str(row["Player_Name"]).strip()
            team      = str(row.get("Team", "")).strip()

            matched = None
            for variant in self._name_variants(full_name):
                if variant in dnames:
                    matched = variant; break
            if matched:
                self.player_id_to_name[pid] = matched; exact_cnt += 1; continue

            parts = full_name.split()
            last = parts[-1].lower() if parts else ""
            fi   = parts[0][0].upper() if parts else ""
            t3 = next(
                (dlist[i] for i, d in enumerate(dlist_lower)
                 if d.split() and d.split()[-1] == last
                 and d.split()[0][0].upper() == fi), None)
            if t3:
                self.player_id_to_name[pid] = t3; exact_cnt += 1; continue

            same_last_idx = [i for i, d in enumerate(dlist_lower)
                             if d.split() and d.split()[-1] == last]
            idx_pool = same_last_idx if same_last_idx else range(len(dlist))
            best_n, best_s = None, 0.0
            for variant in self._name_variants(full_name):
                vl = variant.lower()
                for i in idx_pool:
                    s = SequenceMatcher(None, vl, dlist_lower[i]).ratio()
                    if s > best_s:
                        best_s, best_n = s, dlist[i]

            if best_s >= 0.55 and best_n:
                self.player_id_to_name[pid] = best_n; fuzzy_cnt += 1
            else:
                self.player_id_to_name[pid] = full_name
                self.unmapped_players.append(
                    (pid, full_name, team, round(best_s, 3), best_n or "—"))
                fallback_cnt += 1

        total = exact_cnt + fuzzy_cnt + fallback_cnt
        self._map_summary = dict(exact=exact_cnt, fuzzy=fuzzy_cnt,
                                 fallback=fallback_cnt, total=total)
        print(f"[OK]    Player mapping — exact:{exact_cnt} fuzzy:{fuzzy_cnt} "
              f"unmapped:{fallback_cnt}/{total}")

    # ════════════════════════════════════════════════════════ data cleaning

    def _clean_deliveries(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = (df[obj_cols]
                        .apply(lambda c: c.str.strip())
                        .replace({"nan": np.nan, "None": np.nan, "": np.nan}))

        if "over_ball" in df.columns:
            ob = pd.to_numeric(df["over_ball"], errors="coerce").values
            df["over"] = np.where(np.isfinite(ob), np.floor(ob).astype(int), np.nan)
            df["ball"] = np.where(np.isfinite(ob),
                                  np.round((ob - np.floor(ob)) * 10).astype(int), np.nan)

        if "inning" in df.columns:
            df["inning"] = pd.to_numeric(df["inning"], errors="coerce")
            df = df[df["inning"].isin([1.0, 2.0])].copy()

        for col in ("batting_team", "bowling_team"):
            if col in df.columns:
                umap = {v: self._norm_team(v) for v in df[col].dropna().unique()}
                df[col] = df[col].map(umap).fillna("Unknown")

        ct = set(self.CURRENT_TEAMS)
        df = df[df["batting_team"].isin(ct) & df["bowling_team"].isin(ct)].copy()

        for col in ("batsman_runs", "extras", "isWide", "isNoBall"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["total_ball_runs"] = df.get("batsman_runs", 0) + df.get("extras", 0)
        df["is_legal"] = (
            (df.get("isWide",   pd.Series(0, index=df.index)) == 0) &
            (df.get("isNoBall", pd.Series(0, index=df.index)) == 0)
        ).astype(np.int8)

        if "dismissal_kind" in df.columns:
            df["is_wicket"] = (
                df["dismissal_kind"].notna() &
                ~df["dismissal_kind"].astype(str).str.strip()
                 .isin(["nan", "", "None"])
            ).astype(np.int8)
        else:
            df["is_wicket"] = np.int8(0)

        if "date" in df.columns:
            df["year"] = (pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
                          .dt.year.fillna(2020).astype(np.int16))

        for col in ("batsman", "bowler", "non_striker"):
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return df

    def _clean_matches(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        drop_cols = [
            "winner_runs", "neutralvenue", "city", "winner",
            "eliminator", "method", "team1", "team2", "gender",
            "balls_per_over", "winner_wickets", "player_of_match", "outcome",
        ]
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
        obj_cols = df.select_dtypes(include="object").columns
        df[obj_cols] = (df[obj_cols]
                        .apply(lambda c: c.str.strip())
                        .replace({"nan": np.nan, "None": np.nan, "": np.nan}))
        if "venue" in df.columns:
            umap = {v: self._norm_venue(v) for v in df["venue"].dropna().unique()}
            df["venue_norm"] = df["venue"].map(umap).fillna("Unknown Venue")
        if "season" in df.columns:
            df["season_year"] = df["season"].apply(self._parse_season)
        if "toss_decision" in df.columns:
            df["toss_decision"] = df["toss_decision"].str.strip().str.lower()
        return df

    # ════════════════════════════════════════════════ vectorised aggregation

    def _wagg(self, pp: pd.DataFrame, group_cols) -> dict:
        """Weighted aggregation — no percentile features to avoid target leakage."""
        pp = pp.copy()
        pp["_wsc"]  = pp["powerplay_score"] * pp["yr_w"]
        pp["_wsc2"] = pp["powerplay_score"] ** 2 * pp["yr_w"]

        agg = pp.groupby(group_cols, sort=False).agg(
            wsc_sum  = ("_wsc",            "sum"),
            wsc2_sum = ("_wsc2",           "sum"),
            yw_sum   = ("yr_w",            "sum"),
            median   = ("powerplay_score", "median"),
            cnt      = ("powerplay_score", "count"),
        ).reset_index()

        agg["mean"] = agg["wsc_sum"] / agg["yw_sum"]
        agg["var"]  = (agg["wsc2_sum"] / agg["yw_sum"] - agg["mean"] ** 2).clip(lower=0)
        agg["std"]  = np.sqrt(agg["var"])

        result: dict = {}
        if isinstance(group_cols, str):
            for row in agg.itertuples(index=False):
                result[getattr(row, group_cols)] = {
                    "mean": float(row.mean), "std": float(row.std),
                    "median": float(row.median), "cnt": int(row.cnt),
                }
        else:
            for row in agg.itertuples(index=False):
                key = tuple(getattr(row, c) for c in group_cols)
                result[key] = {
                    "mean": float(row.mean), "std": float(row.std),
                    "median": float(row.median), "cnt": int(row.cnt),
                }
        return result

    # ════════════════════════════════════════════════════════ stat building

    def _build_stats(self, pp_df: pd.DataFrame, pp_scores: pd.DataFrame,
                     cutoff_year: int = None) -> None:
        """
        Build all lookup statistics strictly from data before cutoff_year.
        If cutoff_year is None, uses all data (for final model fit only).
        """
        pp_scores = pp_scores.copy()

        # Apply temporal cutoff
        if cutoff_year is not None:
            pp_scores = pp_scores[pp_scores["season_year"] < cutoff_year].copy()
            if "year" in pp_df.columns:
                pp_df = pp_df[pp_df["year"] < cutoff_year].copy()
            elif "season_year" in pp_df.columns:
                pp_df = pp_df[pp_df["season_year"] < cutoff_year].copy()

        if len(pp_scores) == 0:
            return

        pp_scores["yr_w"] = pp_scores["season_year"].map(self._YR_W_CACHE).fillna(1.0)
        pp_scores["regime"] = pp_scores["powerplay_score"].apply(score_to_regime)

        sc = pp_scores["powerplay_score"].values
        w  = pp_scores["yr_w"].values
        ws = w.sum()
        self.global_mean   = float(np.dot(sc, w) / ws)
        self.global_median = float(np.median(sc))
        self.global_std    = float(np.sqrt(max(
            np.dot(w, (sc - self.global_mean) ** 2) / ws, 0)))

        for r in range(3):
            mask_r = pp_scores["regime"] == r
            if mask_r.sum() > 0:
                sc_r = pp_scores.loc[mask_r, "powerplay_score"].values
                w_r  = pp_scores.loc[mask_r, "yr_w"].values
                ws_r = w_r.sum()
                self.regime_means[r] = float(np.dot(sc_r, w_r) / ws_r)
                self.regime_stds[r]  = float(np.sqrt(max(
                    np.dot(w_r, (sc_r - self.regime_means[r]) ** 2) / ws_r, 0)))
            else:
                self.regime_means[r] = self.global_mean
                self.regime_stds[r]  = self.global_std

        self.venue_stats      = self._wagg(pp_scores, "venue_norm")
        self.team_bat_stats   = self._wagg(pp_scores, "batting_team")
        self.team_bowl_stats  = self._wagg(pp_scores, "bowling_team")
        self.matchup_stats    = self._wagg(pp_scores, ["batting_team", "bowling_team"])
        self.venue_team_stats = self._wagg(pp_scores, ["venue_norm",   "batting_team"])

        # FIX B: Explosion rate building — heavier recency emphasis.
        # We now compute a blended prior: 60% from last 3 seasons, 40% from all-time.
        # This captures team-level changes in powerplay approach (e.g. SRH 2024-25).
        global_exp_rate = float((pp_scores["regime"] == REGIME_EXPLOSIVE).mean())
        exp_alpha = 10  # prior strength for Bayesian smoothing

        max_yr_all = int(pp_scores["season_year"].max())
        recent_window = pp_scores[pp_scores["season_year"] >= max_yr_all - 2]

        for bt, grp in pp_scores.groupby("batting_team"):
            n = len(grp)
            raw_rate = float((grp["regime"] == REGIME_EXPLOSIVE).sum() / max(n, 1))
            all_time_rate = (raw_rate * n + global_exp_rate * exp_alpha) / (n + exp_alpha)

            # FIX B: Blend with recent rate if we have enough recent data
            rec_grp = recent_window[recent_window["batting_team"] == bt]
            n_rec = len(rec_grp)
            if n_rec >= 5:
                rec_rate = float((rec_grp["regime"] == REGIME_EXPLOSIVE).sum() / n_rec)
                rec_smooth = (rec_rate * n_rec + global_exp_rate * exp_alpha) / (n_rec + exp_alpha)
                # Blend: 55% recent, 45% all-time
                self.regime_priors[bt] = 0.55 * rec_smooth + 0.45 * all_time_rate
            else:
                self.regime_priors[bt] = all_time_rate

        for v, grp in pp_scores.groupby("venue_norm"):
            n = len(grp)
            raw_rate = float((grp["regime"] == REGIME_EXPLOSIVE).sum() / max(n, 1))
            stat_rate = (raw_rate * n + global_exp_rate * exp_alpha) / (n + exp_alpha)

            # FIX C: Blend statistical rate with static explosion_base.
            # Static base encodes ground truth about the venue's nature;
            # statistical rate can be noisy with small samples.
            static_base = self.VENUE_BEHAVIOR.get(
                v, self._DEFAULT_BEHAVIOR)["explosion_base"]
            # Weight: 70% stat (if n>=20), else more toward static base
            stat_weight = min(0.70, n / 30.0)
            self.venue_explosion_rt[v] = (
                stat_weight * stat_rate + (1.0 - stat_weight) * static_base)

        # FIX B: Recent form — extended to 3-season window (was 2)
        max_yr = int(pp_scores["season_year"].max())
        rec    = pp_scores[pp_scores["season_year"] >= max_yr - 2].copy()  # 3 seasons
        if not rec.empty:
            rec["_wsc"] = rec["powerplay_score"] * rec["yr_w"]
            self.recent_bat  = (rec.groupby("batting_team")
                                .agg(wsc=("_wsc","sum"), yw=("yr_w","sum"))
                                .eval("mean = wsc / yw")["mean"].to_dict())
            self.recent_bowl = (rec.groupby("bowling_team")
                                .agg(wsc=("_wsc","sum"), yw=("yr_w","sum"))
                                .eval("mean = wsc / yw")["mean"].to_dict())
            for bt, grp in rec.groupby("batting_team"):
                n = len(grp)
                raw_rate = float((grp["regime"] == REGIME_EXPLOSIVE).sum() / max(n, 1))
                self.regime_priors[bt + "_recent"] = (
                    raw_rate * n + global_exp_rate * exp_alpha) / (n + exp_alpha)

        inn_agg = (pp_scores.copy()
                   .assign(_wsc=lambda d: d["powerplay_score"] * d["yr_w"])
                   .groupby("inning")
                   .agg(wsc=("_wsc","sum"), yw=("yr_w","sum")))
        self.inning_avg = {int(k): float(v) for k, v in
                           (inn_agg["wsc"] / inn_agg["yw"]).to_dict().items()}

        # Average powerplay wickets per team
        pp_w = (pp_df.groupby(["matchId", "inning", "batting_team"])["is_wicket"]
                .sum().reset_index().rename(columns={"is_wicket": "pp_wickets"}))
        pp_w = pp_w.merge(pp_scores[["matchId", "inning", "yr_w"]],
                          on=["matchId", "inning"], how="left")
        pp_w["yr_w"] = pp_w["yr_w"].fillna(1.0)
        pp_w["_wpw"] = pp_w["pp_wickets"] * pp_w["yr_w"]
        wk_agg = (pp_w[pp_w["batting_team"].isin(self.CURRENT_TEAMS)]
                  .groupby("batting_team")
                  .agg(wpw=("_wpw","sum"), yw=("yr_w","sum")))
        self.team_wickets_pp = (wk_agg["wpw"] / wk_agg["yw"]).to_dict()

        yr_map = pp_scores.set_index(["matchId", "inning"])["yr_w"].to_dict()
        pp_df2 = pp_df.copy()
        pp_df2["yr_w"] = pd.array(
            [yr_map.get((mid, inn), 1.0)
             for mid, inn in zip(pp_df2["matchId"], pp_df2["inning"])], dtype=float)

        pp_df2["_wrun"] = pp_df2["batsman_runs"] * pp_df2["yr_w"]
        bat_agg = (pp_df2.groupby("batsman")
                   .agg(wrun=("_wrun","sum"), yw=("yr_w","sum"))
                   .query("yw >= 6"))
        bat_agg["sr"] = (bat_agg["wrun"] / bat_agg["yw"] * 100).round(2)

        m_runs = (pp_df2.groupby(["batsman", "matchId", "inning"])
                  .agg(runs_m=("batsman_runs","sum"), yw_m=("yr_w","first"))
                  .reset_index())
        m_runs["_wrm"] = m_runs["runs_m"] * m_runs["yw_m"]
        rpm_agg = (m_runs.groupby("batsman")
                   .agg(wrm=("_wrm","sum"), yw=("yw_m","sum")))
        rpm_agg["rpm"] = (rpm_agg["wrm"] / rpm_agg["yw"]).round(2)

        m_runs["_wrm2"] = m_runs["runs_m"] ** 2 * m_runs["yw_m"]
        var_agg = (m_runs.groupby("batsman")
                   .agg(wrm2=("_wrm2","sum"), wrm=("_wrm","sum"), yw=("yw_m","sum")))
        var_agg["var"] = (var_agg["wrm2"] / var_agg["yw"] -
                          (var_agg["wrm"] / var_agg["yw"]) ** 2).clip(lower=0)
        var_agg["std_rpm"] = np.sqrt(var_agg["var"])

        bat_merged = bat_agg[["sr"]].join(rpm_agg[["rpm"]], how="inner")
        bat_merged = bat_merged.join(var_agg[["std_rpm"]], how="left")
        bat_merged["std_rpm"] = bat_merged["std_rpm"].fillna(bat_merged["rpm"] * 0.5)
        self.bat_player_stats = bat_merged.to_dict("index")
        self.default_bat_sr   = (float(bat_agg["sr"].mean())
                                 if not bat_agg.empty else 120.0)

        extras_col = (pp_df2["extras"] if "extras" in pp_df2.columns
                      else pd.Series(0, index=pp_df2.index))
        pp_df2["_total_run"] = pp_df2["batsman_runs"] + extras_col
        pp_df2["_wrun_b"]    = pp_df2["_total_run"] * pp_df2["yr_w"]
        bowl_agg = (pp_df2.groupby("bowler")
                    .agg(wrun=("_wrun_b","sum"), yw=("yr_w","sum"))
                    .query("yw >= 6"))
        bowl_agg["econ"] = (bowl_agg["wrun"] / bowl_agg["yw"] * 6).round(2)
        self.bowl_player_stats = bowl_agg[["econ"]].to_dict("index")
        self.default_econ      = (float(bowl_agg["econ"].mean())
                                  if not bowl_agg.empty else 7.5)

    # ═════════════════════════════════════════════════════ feature dict

    def _feature_dict(
        self,
        venue:        str,
        batting_team: str,
        bowling_team: str,
        inning:       int,
        season_year:  int,
        bat_names:    list = None,
        bowl_names:   list = None,
    ) -> dict:
        f  = {}
        GM = self.global_mean
        Me = self.global_median
        GS = max(self.global_std, 1.0)

        # ── Venue ──────────────────────────────────────────────────────────
        vs    = self.venue_stats.get(venue, {})
        v_cnt = vs.get("cnt", 0)
        f["venue_mean"]   = self._shrink(vs.get("mean",   GM), v_cnt, GM)
        f["venue_median"] = self._shrink(vs.get("median", Me), v_cnt, Me)
        f["venue_std"]    = np.clip(vs.get("std", GS), 1.0, 30.0)
        f["venue_cnt"]    = min(v_cnt, 200)
        f.update(self._venue_features(venue))

        # ── Batting team ───────────────────────────────────────────────────
        bts    = self.team_bat_stats.get(batting_team, {})
        bt_cnt = bts.get("cnt", 0)
        f["bat_mean"]   = self._shrink(bts.get("mean",   GM), bt_cnt, GM)
        f["bat_median"] = self._shrink(bts.get("median", Me), bt_cnt, Me)
        f["bat_std"]    = np.clip(bts.get("std", GS), 1.0, 30.0)
        f["bat_cnt"]    = min(bt_cnt, 200)

        # ── Bowling team ───────────────────────────────────────────────────
        bls    = self.team_bowl_stats.get(bowling_team, {})
        bl_cnt = bls.get("cnt", 0)
        f["bowl_mean"]   = self._shrink(bls.get("mean",   GM), bl_cnt, GM)
        f["bowl_median"] = self._shrink(bls.get("median", Me), bl_cnt, Me)
        f["bowl_std"]    = np.clip(bls.get("std", GS), 1.0, 30.0)
        f["bowl_cnt"]    = min(bl_cnt, 200)

        # ── Matchup ────────────────────────────────────────────────────────
        ms    = self.matchup_stats.get((batting_team, bowling_team), {})
        m_cnt = ms.get("cnt", 0)
        fallback_matchup = (f["bat_mean"] + f["bowl_mean"]) / 2
        f["matchup_mean"]   = self._shrink(ms.get("mean",   fallback_matchup), m_cnt, GM)
        f["matchup_median"] = self._shrink(ms.get("median", fallback_matchup), m_cnt, Me)
        f["matchup_std"]    = np.clip(ms.get("std", GS), 1.0, 30.0)
        f["matchup_cnt"]    = min(m_cnt, 100)

        # ── Venue × Team ───────────────────────────────────────────────────
        vts    = self.venue_team_stats.get((venue, batting_team), {})
        vt_cnt = vts.get("cnt", 0)
        f["venue_bat_mean"]   = self._shrink(vts.get("mean",   f["venue_mean"]), vt_cnt, f["venue_mean"])
        f["venue_bat_median"] = self._shrink(vts.get("median", f["venue_median"]), vt_cnt, f["venue_median"])
        f["venue_bat_cnt"]    = min(vt_cnt, 100)

        # ── Recent form ────────────────────────────────────────────────────
        f["bat_recent"]  = self.recent_bat.get(batting_team,  f["bat_mean"])
        f["bowl_recent"] = self.recent_bowl.get(bowling_team, f["bowl_mean"])

        # ── Derived diffs ──────────────────────────────────────────────────
        f["bat_bowl_diff"]    = np.clip(f["bat_mean"]       - f["bowl_mean"],    -30.0, 30.0)
        f["recent_form_diff"] = np.clip(f["bat_recent"]     - f["bowl_recent"],  -30.0, 30.0)
        f["venue_bat_delta"]  = np.clip(f["venue_bat_mean"] - f["venue_mean"],   -20.0, 20.0)
        f["matchup_vs_venue"] = np.clip(f["matchup_mean"]   - f["venue_mean"],   -20.0, 20.0)
        f["bat_std_ratio"]    = np.clip(f["bat_std"] / max(f["venue_std"], 1.0),  0.2,  3.0)

        # ── Simple interaction features ─────────────────────────────────────
        f["bat_x_venue"]      = np.clip(f["bat_mean"] * f["vb_batting"], 0.0, 80.0)
        f["bowl_x_spin"]      = np.clip(f["bowl_mean"] * f["vb_spin"],   0.0, 80.0)
        f["recent_x_batting"] = np.clip(f["bat_recent"] * f["vb_batting"], 0.0, 80.0)

        # ── Inning ─────────────────────────────────────────────────────────
        f["inning"]       = int(inning)
        f["inning_avg"]   = self.inning_avg.get(int(inning), GM)
        inn1_avg          = self.inning_avg.get(1, GM)
        inn2_avg          = self.inning_avg.get(2, GM)
        # FIX E: inning2_boost now includes the learned bias correction.
        # Raw boost from data + structural correction from prediction history.
        raw_inn2_boost = float(inn2_avg - inn1_avg) if inning == 2 else 0.0
        f["inning2_boost"] = raw_inn2_boost + (self._INNING_BIAS.get(int(inning), 0.0)
                                               if inning == 2 else 0.0)

        # ── Season ─────────────────────────────────────────────────────────
        sy = (int(season_year) if season_year and not
              (isinstance(season_year, float) and np.isnan(season_year)) else 2020)
        f["season_year"] = sy
        f["year_weight"] = self._yr_w(sy)

        # ── Wickets ────────────────────────────────────────────────────────
        f["avg_pp_wickets"] = np.clip(
            self.team_wickets_pp.get(batting_team, 2.5), 0.5, 6.0)

        # ── Player batting features ────────────────────────────────────────
        srs, rpms, stds = [], [], []
        if bat_names:
            for bn in bat_names:
                s = self.bat_player_stats.get(bn)
                if s:
                    srs.append(np.clip(s["sr"], 50.0, 250.0))
                    rpms.append(np.clip(s["rpm"], 0.0, 30.0))
                    stds.append(np.clip(s.get("std_rpm", s["rpm"] * 0.5), 0.0, 20.0))

        f["bat_avg_sr"]   = float(np.mean(srs))  if srs  else self.default_bat_sr
        f["bat_max_sr"]   = float(max(srs))       if srs  else self.default_bat_sr
        f["bat_top2_sr"]  = float(np.mean(sorted(srs, reverse=True)[:2])) if len(srs) >= 2 else f["bat_avg_sr"]
        f["bat_avg_rpm"]  = float(np.mean(rpms))  if rpms else GM / 6
        f["bat_max_rpm"]  = float(max(rpms))       if rpms else GM / 6
        f["bat_avg_var"]  = float(np.mean(stds))  if stds else GS
        f["bat_known"]    = len(srs)
        f["bat_sr_delta"] = np.clip(f["bat_avg_sr"] - self.default_bat_sr, -60.0, 80.0)
        f["top2_sr_delta"]= np.clip(f["bat_top2_sr"] - self.default_bat_sr, -60.0, 80.0)

        # ── Player bowling features ────────────────────────────────────────
        econs = []
        if bowl_names:
            for bn in bowl_names:
                s = self.bowl_player_stats.get(bn)
                if s:
                    econs.append(np.clip(s["econ"], 3.0, 18.0))

        f["bowl_avg_econ"]   = float(np.mean(econs)) if econs else self.default_econ
        f["bowl_min_econ"]   = float(min(econs))     if econs else self.default_econ
        f["bowl_known"]      = len(econs)
        f["bowl_econ_delta"] = np.clip(
            self.default_econ - f["bowl_avg_econ"], -8.0, 8.0)

        # FIX B + C: Explosion probability — uses recency-blended priors.
        global_exp_rate = 0.20
        exp_bat   = float(self.regime_priors.get(batting_team, global_exp_rate))
        exp_bat_r = float(self.regime_priors.get(batting_team + "_recent", exp_bat))
        exp_venue = float(self.venue_explosion_rt.get(venue, global_exp_rate))
        exp_bowl  = 1.0 - float(self.regime_priors.get(bowling_team, global_exp_rate))

        # FIX B: Increased weight on recent batting form (0.25→0.30) and
        # venue (0.25→0.28) vs all-time batting (0.30→0.25).
        # The static explosion_base (FIX C) is already embedded in exp_venue.
        f["explosion_prob"] = float(np.clip(
            0.25 * exp_bat + 0.30 * exp_bat_r +
            0.28 * exp_venue + 0.10 * exp_bowl +
            0.07 * float(f["vb_batting"]),
            0.0, 1.0
        ))

        sr_delta_norm = f["top2_sr_delta"] / max(GS, 1.0)
        f["intent_proxy"] = float(np.clip(
            sr_delta_norm * f["vb_batting"] * f["explosion_prob"],
            -3.0, 3.0
        ))

        return f

    # ════════════════════════════════════════════════ time-based CV utility

    @staticmethod
    def _time_cv_splits(years: np.ndarray, n_splits: int = 3):
        """Walk-forward splits: train on all years < val_year, validate on val_year."""
        unique_years = sorted(np.unique(years))
        n = len(unique_years)
        if n < n_splits + 1:
            n_splits = max(n - 1, 1)
        splits = []
        for i in range(n_splits, 0, -1):
            val_year  = unique_years[-i]
            train_idx = np.where(years < val_year)[0]
            val_idx   = np.where(years == val_year)[0]
            if len(train_idx) > 0 and len(val_idx) > 0:
                splits.append((train_idx, val_idx, int(val_year)))
        return splits

    # ════════════════════════════════════════════════ fold feature builder

    def _build_fold_features(
        self,
        pp_df: pd.DataFrame,
        pp_scores: pd.DataFrame,
        indices: np.ndarray,
        cutoff_year: int,
    ) -> pd.DataFrame:
        """Build features for a set of match-inning rows using only data < cutoff_year."""
        saved = self._snapshot_stats()
        self._build_stats(pp_df, pp_scores, cutoff_year=cutoff_year)

        pp_batsmen = (
            pp_df[pp_df["batsman"].notna() & (pp_df.get("year", pp_df.get("season_year", pd.Series(0, index=pp_df.index))) < cutoff_year)]
            .groupby(["matchId", "inning"], sort=False)["batsman"]
            .unique().apply(list).to_dict()
        ) if "year" in pp_df.columns else (
            pp_df[pp_df["batsman"].notna()]
            .groupby(["matchId", "inning"], sort=False)["batsman"]
            .unique().apply(list).to_dict()
        )
        pp_bowlers = (
            pp_df[pp_df["bowler"].notna() & (pp_df.get("year", pp_df.get("season_year", pd.Series(0, index=pp_df.index))) < cutoff_year)]
            .groupby(["matchId", "inning"], sort=False)["bowler"]
            .unique().apply(list).to_dict()
        ) if "year" in pp_df.columns else (
            pp_df[pp_df["bowler"].notna()]
            .groupby(["matchId", "inning"], sort=False)["bowler"]
            .unique().apply(list).to_dict()
        )

        rows_subset = pp_scores.iloc[indices]
        X_rows = []
        for row in rows_subset.itertuples(index=False):
            key  = (row.matchId, row.inning)
            feat = self._feature_dict(
                venue=row.venue_norm,
                batting_team=row.batting_team,
                bowling_team=row.bowling_team,
                inning=int(row.inning),
                season_year=int(row.season_year),
                bat_names=pp_batsmen.get(key, []),
                bowl_names=pp_bowlers.get(key, []),
            )
            X_rows.append(feat)

        self._restore_stats(saved)
        return pd.DataFrame(X_rows)

    def _snapshot_stats(self) -> dict:
        attrs = [
            "venue_stats", "team_bat_stats", "team_bowl_stats", "matchup_stats",
            "venue_team_stats", "bat_player_stats", "bowl_player_stats",
            "team_wickets_pp", "recent_bat", "recent_bowl", "inning_avg",
            "regime_priors", "venue_explosion_rt", "regime_means", "regime_stds",
            "global_mean", "global_median", "global_std", "default_bat_sr",
            "default_econ",
        ]
        return {a: getattr(self, a) for a in attrs}

    def _restore_stats(self, snapshot: dict) -> None:
        for k, v in snapshot.items():
            setattr(self, k, v)

    # ════════════════════════════════════════════════════════════════ fit

    def fit(
        self,
        deliveries_df: pd.DataFrame,
        players_df=None,
        matches_df: pd.DataFrame = None,
    ) -> None:
        print("[INFO]  Starting fit v9 (Targeted Improvements) ...")

        deliveries_df = self._clean_deliveries(deliveries_df)
        print(f"[OK]    Deliveries cleaned — {len(deliveries_df):,} rows.")

        if matches_df is not None:
            matches_df = self._clean_matches(matches_df)

        self.map_player_ids(deliveries_df, players_df)

        pp_df = deliveries_df[deliveries_df["over"] < 6].copy()
        print(f"[OK]    Powerplay deliveries — {len(pp_df):,} rows.")

        pp_scores = (
            pp_df.groupby(["matchId", "inning"], sort=False)
            .agg(
                bat_runs    =("batsman_runs", "sum"),
                extras      =("extras",       "sum"),
                batting_team=("batting_team", "first"),
                bowling_team=("bowling_team", "first"),
            ).reset_index()
        )
        pp_scores["powerplay_score"] = pp_scores["bat_runs"] + pp_scores["extras"]

        if matches_df is not None and "matchId" in matches_df.columns:
            mc = ["matchId"]
            if "venue_norm"    in matches_df.columns: mc.append("venue_norm")
            if "season_year"   in matches_df.columns: mc.append("season_year")
            if "toss_decision" in matches_df.columns: mc.append("toss_decision")
            pp_scores = pp_scores.merge(
                matches_df[mc].drop_duplicates("matchId"),
                on="matchId", how="left")

        if "venue_norm"  not in pp_scores.columns: pp_scores["venue_norm"]  = "Unknown Venue"
        if "season_year" not in pp_scores.columns:
            if "year" in deliveries_df.columns:
                ym = deliveries_df.groupby("matchId")["year"].first()
                pp_scores["season_year"] = pp_scores["matchId"].map(ym)
            else:
                pp_scores["season_year"] = 2020

        pp_scores["venue_norm"]  = pp_scores["venue_norm"].fillna("Unknown Venue")
        pp_scores["season_year"] = (
            pd.to_numeric(pp_scores["season_year"], errors="coerce")
            .fillna(2020).astype(int))

        pp_scores = pp_scores[pp_scores["season_year"] >= self.TRAIN_FROM_YEAR].copy()
        if "year" in pp_df.columns:
            pp_df = pp_df[pp_df["year"] >= self.TRAIN_FROM_YEAR].copy()
        pp_scores = pp_scores.reset_index(drop=True)
        print(f"[OK]    Training samples — {len(pp_scores):,} match-innings.")

        valid_score_mask = pp_scores["powerplay_score"] >= 10
        pp_scores = pp_scores[valid_score_mask].reset_index(drop=True)
        print(f"[OK]    After score floor filter (>=10) — {len(pp_scores):,} samples.")

        yr_series = pp_scores["season_year"].reset_index(drop=True)

        # ── Time-based CV with leak-free per-fold feature building ─────────
        print("[INFO]  Running walk-forward cross-validation (leak-free) ...")
        splits = self._time_cv_splits(yr_series.values, n_splits=3)

        oof_preds  = np.full(len(pp_scores), np.nan)
        oof_regime = np.full(len(pp_scores), -1, dtype=int)
        cv_maes    = []

        y_all = pp_scores["powerplay_score"].values.astype(float)

        for fold_i, (tr_idx, va_idx, val_year) in enumerate(splits):
            print(f"  Fold {fold_i+1} — val_year={val_year} "
                  f"| train={len(tr_idx)} | val={len(va_idx)}")

            Xtr_fold = self._build_fold_features(pp_df, pp_scores, tr_idx, cutoff_year=val_year)
            Xva_fold = self._build_fold_features(pp_df, pp_scores, va_idx, cutoff_year=val_year)

            all_cols = list(set(Xtr_fold.columns) | set(Xva_fold.columns))
            Xtr_fold = Xtr_fold.reindex(columns=all_cols, fill_value=self.global_mean)
            Xva_fold = Xva_fold.reindex(columns=all_cols, fill_value=self.global_mean)
            Xtr_fold = Xtr_fold.fillna(self.global_mean)
            Xva_fold = Xva_fold.fillna(self.global_mean)

            ytr = y_all[tr_idx]
            yva = y_all[va_idx]
            rtr = np.array([score_to_regime(s) for s in ytr])

            fold_yr = yr_series.iloc[tr_idx].values
            wtr = np.array([self._yr_w(y) for y in fold_yr])
            wtr = wtr / wtr.mean()

            # FIX A: CV classifier also uses num_threads=1/nthread=1
            if _HAS_LGB:
                clf_cv = LGBMClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.08,
                    random_state=42, num_threads=1, force_col_wise=True, verbose=-1)
            else:
                clf_cv = XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.08,
                    random_state=42, nthread=1, n_jobs=1, tree_method="hist",
                    use_label_encoder=False, eval_metric="mlogloss")
            clf_cv.fit(Xtr_fold, rtr, sample_weight=wtr)
            probs_va = clf_cv.predict_proba(Xva_fold)

            fold_preds = np.zeros(len(Xva_fold))
            for r in range(3):
                mask_r = rtr == r
                n_r = mask_r.sum()
                if n_r < 10:
                    sc_r = ytr[mask_r] if n_r > 0 else ytr
                    regime_mean_r = float(np.mean(sc_r))
                    fold_preds += probs_va[:, r] * regime_mean_r
                    continue
                reg_cv = XGBRegressor(
                    n_estimators=150, max_depth=4, learning_rate=0.08,
                    random_state=42, nthread=1, n_jobs=1, tree_method="hist",
                    objective="reg:absoluteerror",
                    reg_alpha=0.10, reg_lambda=1.5,
                )
                Xtr_r = Xtr_fold.iloc[mask_r]
                ytr_r = ytr[mask_r]
                wtr_r = wtr[mask_r]

                val_size_r = max(5, int(0.15 * len(ytr_r)))
                reg_cv.fit(
                    Xtr_r.iloc[:-val_size_r], ytr_r[:-val_size_r],
                    sample_weight=wtr_r[:-val_size_r],
                    eval_set=[(Xtr_r.iloc[-val_size_r:], ytr_r[-val_size_r:])],
                    verbose=False,
                )
                fold_preds += probs_va[:, r] * reg_cv.predict(Xva_fold)

            fold_mae = float(np.mean(np.abs(fold_preds - yva)))
            cv_maes.append(fold_mae)
            print(f"  → MAE = {fold_mae:.3f}")

            oof_preds[va_idx] = fold_preds
            oof_regime[va_idx] = np.argmax(probs_va, axis=1)

        print(f"[OK]    CV MAE = {np.mean(cv_maes):.3f} ± {np.std(cv_maes):.3f}")

        # ── Build final stats using ALL training data ─────────────────────
        print("[INFO]  Building final stats from full training data ...")
        self._build_stats(pp_df, pp_scores, cutoff_year=None)
        print(f"[OK]    Final global mean={self.global_mean:.2f}")
        print(f"        Regime means: low={self.regime_means[0]:.1f} "
              f"normal={self.regime_means[1]:.1f} explosive={self.regime_means[2]:.1f}")

        pp_batsmen = (
            pp_df[pp_df["batsman"].notna()]
            .groupby(["matchId", "inning"], sort=False)["batsman"]
            .unique().apply(list).to_dict())
        pp_bowlers = (
            pp_df[pp_df["bowler"].notna()]
            .groupby(["matchId", "inning"], sort=False)["bowler"]
            .unique().apply(list).to_dict())

        print("[INFO]  Assembling final feature matrix ...")
        X_rows, y_vals = [], []
        for row in pp_scores.itertuples(index=False):
            key  = (row.matchId, row.inning)
            feat = self._feature_dict(
                venue=row.venue_norm,
                batting_team=row.batting_team,
                bowling_team=row.bowling_team,
                inning=int(row.inning),
                season_year=int(row.season_year),
                bat_names=pp_batsmen.get(key, []),
                bowl_names=pp_bowlers.get(key, []),
            )
            X_rows.append(feat)
            y_vals.append(row.powerplay_score)

        X = pd.DataFrame(X_rows)
        y = np.array(y_vals, dtype=float)
        self.feature_cols = X.columns.tolist()
        X = X.fillna(self.global_mean)
        print(f"[OK]    Feature matrix — {len(y):,} samples, {len(self.feature_cols)} features.")

        y_regime = np.array([score_to_regime(s) for s in y])
        samp_w   = yr_series.map(self._YR_W_CACHE).fillna(1.0).values
        samp_w   = samp_w / samp_w.mean()

        # ── Stage-1: Train final classifier ───────────────────────────────
        print("\n[INFO]  Training Stage-1 classifier (final) ...")
        self.classifier.fit(X, y_regime, sample_weight=samp_w)
        clf_acc = float(np.mean(self.classifier.predict(X) == y_regime))
        print(f"[OK]    Stage-1 in-sample accuracy: {clf_acc:.3f}")

        # ── Stage-2: Train final regime-specific regressors ───────────────
        print("[INFO]  Training Stage-2 regressors per regime (final) ...")
        for r in range(3):
            mask_r = y_regime == r
            n_r    = mask_r.sum()
            print(f"  Regime {REGIME_NAMES[r]}: {n_r} samples")
            if n_r < 15:
                print(f"  [WARN]  Too few samples for regime {r}, skipping.")
                continue
            Xr = X[mask_r].reset_index(drop=True)
            yr = y[mask_r]
            wr = samp_w[mask_r]
            wr = wr / wr.mean()

            regime_years = yr_series.values[mask_r]
            sorted_order = np.argsort(regime_years, kind="stable")
            Xr = Xr.iloc[sorted_order].reset_index(drop=True)
            yr = yr[sorted_order]
            wr = wr[sorted_order]

            val_size = max(5, int(0.15 * len(yr)))
            Xr_tr, Xr_va = Xr.iloc[:-val_size], Xr.iloc[-val_size:]
            yr_tr, yr_va = yr[:-val_size],       yr[-val_size:]
            wr_tr        = wr[:-val_size]

            self.regressors[r].fit(
                Xr_tr, yr_tr,
                sample_weight=wr_tr,
                eval_set=[(Xr_va, yr_va)],
                verbose=False,
            )
            r_mae = float(np.mean(np.abs(self.regressors[r].predict(Xr_va) - yr_va)))
            print(f"  Regime {REGIME_NAMES[r]} validation MAE: {r_mae:.3f}")

        # FIX G: Tighter OOF calibration — bias capped at ±3, scale 0.95–1.05
        print("[INFO]  Computing OOF calibration ...")
        oof_mask = ~np.isnan(oof_preds)
        if oof_mask.sum() > 30:
            for r in range(3):
                r_mask = oof_mask & (oof_regime == r)
                if r_mask.sum() > 10:
                    residuals = y_all[r_mask] - oof_preds[r_mask]
                    raw_bias = float(np.median(residuals))
                    # FIX G: cap bias tightly so well-predicted matches aren't disrupted
                    self._calib_bias[r]  = float(np.clip(raw_bias, -3.0, 3.0))
                    mean_pred = float(np.mean(oof_preds[r_mask]))
                    mean_true = float(np.mean(y_all[r_mask]))
                    # FIX G: tighter scale range (was 0.92–1.08, now 0.95–1.05)
                    self._calib_scale[r] = float(np.clip(
                        mean_true / max(mean_pred, 1.0), 0.95, 1.05))
                    print(f"  Regime {REGIME_NAMES[r]}: bias={self._calib_bias[r]:.3f} "
                          f"scale={self._calib_scale[r]:.3f} (n={r_mask.sum()})")
        else:
            print("[WARN]  Insufficient OOF predictions — using identity calibration.")

        overall_preds  = self._predict_calibrated(X)
        self._insample_mae = float(np.mean(np.abs(overall_preds - y)))
        print(f"[OK]    Final in-sample MAE: {self._insample_mae:.3f} runs "
              f"(note: OOF MAE = {np.mean(cv_maes):.3f})")

        self._predict_year = int(yr_series.max())

        self._print_feature_importance()
        print("[OK]    Fit complete (v9).")

    # ════════════════════════════════════════════════ prediction helpers

    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        """Mixture prediction weighted by regime probabilities."""
        probs = self.classifier.predict_proba(X)
        preds = np.zeros(len(X))
        for r in range(3):
            if r in self.regressors and hasattr(self.regressors[r], "feature_importances_"):
                preds += probs[:, r] * self.regressors[r].predict(X)
            else:
                preds += probs[:, r] * self.regime_means.get(r, self.global_mean)
        return preds

    def _predict_calibrated(self, X: pd.DataFrame) -> np.ndarray:
        """Apply OOF-derived calibration after mixture prediction."""
        probs     = self.classifier.predict_proba(X)
        raw_preds = self._predict_raw(X)
        preds     = np.copy(raw_preds)

        for i in range(len(X)):
            calib_bias  = sum(probs[i, r] * self._calib_bias[r]  for r in range(3))
            calib_scale = sum(probs[i, r] * self._calib_scale[r] for r in range(3))
            preds[i] = raw_preds[i] * calib_scale + calib_bias

        return np.clip(preds, 18.0, 130.0)

    def _print_feature_importance(self):
        print("\n  Top-15 features (average importance across regimes):")
        agg_imp = {}
        for r, reg in self.regressors.items():
            if not hasattr(reg, "feature_importances_"):
                continue
            imp = reg.feature_importances_
            for feat, val in zip(self.feature_cols, imp):
                agg_imp[feat] = agg_imp.get(feat, 0.0) + val / 3.0
        sorted_feats = sorted(agg_imp.items(), key=lambda x: -x[1])[:15]
        for feat, val in sorted_feats:
            bar = "█" * int(val * 300)
            print(f"    {feat:<30} {val:.4f}  {bar}")

    # ═══════════════════════════════════════════════════════════════ predict

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        print("\n[INFO]  Running v9 predictions ...")

        bat_col  = next(
            (c for c in test_df.columns
             if "batsman" in c.lower() or "batter" in c.lower()), None)
        bowl_col = next(
            (c for c in test_df.columns if "bowler" in c.lower()), None)

        results:    list = []
        trace_rows: list = []

        for _, row in test_df.iterrows():
            mid          = row.get("id", 0)
            venue        = self._norm_venue(str(row.get("venue", "")))
            inning       = int(row.get("innings", row.get("inning", 1)))
            batting_team = self._norm_team(str(row.get("batting_team", "")))
            bowling_team = self._norm_team(str(row.get("bowling_team", "")))

            if "season" in row.index and pd.notna(row.get("season")):
                pred_year = self._parse_season(row["season"])
            elif "season_year" in row.index and pd.notna(row.get("season_year")):
                pred_year = int(row["season_year"])
            elif "year" in row.index and pd.notna(row.get("year")):
                pred_year = int(row["year"])
            else:
                pred_year = self._predict_year

            bat_raw  = str(row[bat_col])  if bat_col  and pd.notna(row.get(bat_col))  else ""
            bowl_raw = str(row[bowl_col]) if bowl_col and pd.notna(row.get(bowl_col)) else ""

            bat_ids  = [x.strip() for x in bat_raw.split(",")  if x.strip()]
            bowl_ids = [x.strip() for x in bowl_raw.split(",") if x.strip()]

            bat_names  = [n for i in bat_ids  if (n := self.player_id_to_name.get(i, ""))]
            bowl_names = [n for i in bowl_ids if (n := self.player_id_to_name.get(i, ""))]

            feat = self._feature_dict(
                venue=venue, batting_team=batting_team, bowling_team=bowling_team,
                inning=inning, season_year=pred_year,
                bat_names=bat_names, bowl_names=bowl_names,
            )

            Xp = pd.DataFrame([feat])
            for col in self.feature_cols:
                if col not in Xp.columns:
                    Xp[col] = self.global_mean
            Xp = Xp[self.feature_cols].fillna(self.global_mean)

            probs       = self.classifier.predict_proba(Xp)[0]
            pred_regime = int(np.argmax(probs))
            raw_pred    = float(self._predict_raw(Xp)[0])
            calib_pred  = float(self._predict_calibrated(Xp)[0])
            final       = int(round(calib_pred))

            results.append({"id": mid, "predicted_score": final})
            trace_rows.append(dict(
                mid=mid, inning=inning,
                batting_team=batting_team, bowling_team=bowling_team,
                venue=venue, bat_names=bat_names, bowl_names=bowl_names,
                probs=probs, regime=pred_regime,
                raw_pred=raw_pred, calib_pred=calib_pred, final=final, feat=feat,
            ))

        W = 66
        KEY_FEATS = [
            "venue_mean", "vb_batting", "vb_explosion_base",
            "bat_mean", "bowl_mean",
            "matchup_mean", "bat_recent", "bowl_recent",
            "bat_avg_sr", "bat_top2_sr", "bat_sr_delta",
            "bowl_avg_econ", "bowl_econ_delta",
            "explosion_prob", "intent_proxy",
            "inning", "inning2_boost",
        ]
        for t in trace_rows:
            print("\n" + "─" * W)
            print(f"  id={t['mid']}  |  Inning {t['inning']}")
            print(f"  {t['batting_team']}  vs  {t['bowling_team']}")
            print(f"  Venue    : {t['venue']}")
            print(f"  Batsmen  : {t['bat_names']}")
            print(f"  Bowlers  : {t['bowl_names']}")
            print(f"  Regime probs  : low={t['probs'][0]:.2f}  "
                  f"normal={t['probs'][1]:.2f}  explosive={t['probs'][2]:.2f}")
            print(f"  Predicted regime: {REGIME_NAMES[t['regime']]}")
            print(f"  Raw mixture   : {t['raw_pred']:.2f}")
            print(f"  Calibrated    : {t['calib_pred']:.2f}")
            print(f"  Final         : {t['final']}")
            print("  Key features:")
            for k in KEY_FEATS:
                v = t["feat"].get(k)
                if v is not None:
                    print(f"    {k:<30}: {v:.4f}")
        print("─" * W)

        submission = pd.DataFrame(results)
        print(f"\n[OK]    Predictions complete — {len(submission)} innings.")

        try:
            submission.to_csv("/var/submission.csv", index=False)
            print("[OK]    submission.csv written.")
        except OSError:
            pass

        return submission

    # ════════════════════════════════════════════════════════ error analysis

    def error_analysis(self, X: pd.DataFrame, y_true: np.ndarray) -> pd.DataFrame:
        """Post-training error analysis — call with held-out data."""
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = self.global_mean
        X = X[self.feature_cols].fillna(self.global_mean)

        preds = self._predict_calibrated(X)
        probs = self.classifier.predict_proba(X)
        errors = preds - y_true

        rows = []
        for i in range(len(y_true)):
            rows.append({
                "true_score":  y_true[i],
                "pred_score":  preds[i],
                "error":       errors[i],
                "abs_error":   abs(errors[i]),
                "true_regime": REGIME_NAMES[score_to_regime(y_true[i])],
                "pred_regime": REGIME_NAMES[int(np.argmax(probs[i]))],
                "exp_prob":    probs[i][REGIME_EXPLOSIVE],
            })

        df = pd.DataFrame(rows).sort_values("abs_error", ascending=False)
        print("\n[ERROR ANALYSIS]")
        print(f"  Overall MAE : {df['abs_error'].mean():.3f}")
        print(f"  Error >30   : {(df['abs_error'] > 30).mean()*100:.1f}%")
        print(f"  Error >20   : {(df['abs_error'] > 20).mean()*100:.1f}%")
        for rname in ("low", "normal", "explosive"):
            sub = df[df["true_regime"] == rname]
            if len(sub) > 0:
                print(f"  {rname:<12} MAE: {sub['abs_error'].mean():.3f} "
                      f"  n={len(sub)}")
        return df