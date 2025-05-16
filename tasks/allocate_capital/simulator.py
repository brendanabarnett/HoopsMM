from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class SimulationSummary:
    expected_return: float
    std_pnl: float
    t_stat: float
    prob_loss: float
    min_pnl: float
    max_pnl: float
    pct25: float
    pct75: float

class MonteCarloSimulator:
    def __init__(self, bet_records):
        self._bets = bet_records  # pd.DataFrame with "pnl", "win", "stake" columns

    def run(self, num_sim: int = 100_000, plot_path: Path | None = None) -> SimulationSummary:
        wins        = self._bets[self._bets.win].shape[0]
        losses      = self._bets.shape[0] - wins
        win_payouts = self._bets.loc[self._bets.win, "pnl"].values
        loss_payouts = self._bets.loc[~self._bets.win, "pnl"].values  # negative

        avg_win, std_win   = win_payouts.mean(), win_payouts.std(ddof=1)
        avg_loss, std_loss = loss_payouts.mean(), loss_payouts.std(ddof=1)
        num_bets = len(self._bets)

        alpha, beta = wins + 1, losses + 1  # Beta prior parameters
        results = np.empty(num_sim)

        for i in range(num_sim):
            p = np.random.beta(alpha, beta)
            outcomes = np.random.rand(num_bets) < p
            wins_raw   = np.random.normal(avg_win,   std_win,   size=num_bets)
            losses_raw = np.random.normal(avg_loss,  std_loss,  size=num_bets)
            wins_pay   = np.clip(wins_raw,   0, None)
            loss_pay   = np.clip(losses_raw, None, 0)
            results[i] = np.where(outcomes, wins_pay, loss_pay).sum()

        mean_pnl  = results.mean()
        std_pnl   = results.std(ddof=1)
        t_stat    = mean_pnl / (std_pnl / np.sqrt(num_bets))
        prob_loss = np.mean(results < 0)

        if plot_path is not None:
            plt.hist(results, bins=50, edgecolor="black", alpha=0.7)
            plt.axvline(mean_pnl, color="red", linestyle="--", label=f"Mean = ${mean_pnl:.2f}")
            plt.xlabel("PNL ($)")
            plt.ylabel("Frequency")
            plt.title("Monte Carlo Distribution")
            plt.legend()
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path)
            plt.close()

        return SimulationSummary(
            expected_return=mean_pnl,
            std_pnl=std_pnl,
            t_stat=t_stat,
            prob_loss=prob_loss,
            min_pnl=results.min(),
            max_pnl=results.max(),
            pct25=np.percentile(results, 25),
            pct75=np.percentile(results, 75),
        )