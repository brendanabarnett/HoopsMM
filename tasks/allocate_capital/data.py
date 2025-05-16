from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import numpy as np

_DECIMAL_EPS = 1e-9  # avoid division by zero


def load_data(path: str | Path) -> pd.DataFrame:
    """Read merged game-level CSV including predictions and betting odds."""
    return pd.read_csv(Path(path))


def american_to_decimal(odds: pd.Series) -> pd.Series:
    """Convert American odds to decimal odds."""
    positive = odds > 0
    dec = odds.copy().astype(float)
    dec[positive] = (dec[positive] / 100) + 1.0
    dec[~positive] = (100 / (np.abs(dec[~positive]) + _DECIMAL_EPS)) + 1.0
    return dec


def build_bet_records(
    df: pd.DataFrame,
    pred_col: str,
    prob_col: str,
    home_fav_col: str,
    fav_odds_col: str,
    dog_odds_col: str,
    stake_strategy: str = "flat",  # "flat" or "kelly"
    flat_stake: float = 100.0,
    bankroll: float = 1_000.0,
) -> pd.DataFrame:
    """Return per-bet record with stake, pnl, win/loss flag.

    * flat stake - constant dollar amount
    * kelly - fractional Kelly criterion per bet
    """
    recs: List[dict] = []
    bankroll_curr = bankroll

    for row in df.itertuples(index=False):
        pred     = getattr(row, pred_col)
        proba    = getattr(row, prob_col)
        actual   = getattr(row, "home_win")  # ground‑truth label
        home_fav = getattr(row, home_fav_col)
        fav_odds = getattr(row, fav_odds_col)
        dog_odds = getattr(row, dog_odds_col)

        fav_dec = american_to_decimal(pd.Series([fav_odds]))[0]
        dog_dec = american_to_decimal(pd.Series([dog_odds]))[0]

        if pred == 1:
            odds_dec = fav_dec if home_fav == 1 else dog_dec
            win_prob = proba
        else:
            odds_dec = dog_dec if home_fav == 1 else fav_dec
            win_prob = 1 - proba

        b = odds_dec - 1.0  # net win multiple
        q = 1.0 - win_prob

        if stake_strategy == "flat":
            stake = flat_stake
        elif stake_strategy == "kelly":
            k_frac = max((b * win_prob - q) / (b + _DECIMAL_EPS), 0)
            stake = k_frac * bankroll_curr
        else:
            raise ValueError(f"Unknown stake_strategy: {stake_strategy}")

        win = ((pred == 1 and actual == 1) or (pred == 0 and actual == 0))
        pnl = stake * b if win else -stake
        bankroll_curr += pnl

        recs.append({
            "stake": stake,
            "pnl": pnl,
            "win": win,
            "odds_dec": odds_dec,
            "bankroll_after": bankroll_curr,
        })

    return pd.DataFrame(recs)
