import argparse
from pathlib import Path
import yaml
from utils.config import load_config
from tasks.allocate_capital.data import load_data, build_bet_records
from tasks.allocate_capital.simulator import MonteCarloSimulator

def main(cfg_path: str):
    cfg = load_config(cfg_path)

    df = load_data(cfg["data"]["path"])
    bet_recs = build_bet_records(
        df=df,
        pred_col   = cfg["data"]["pred_col"],
        prob_col   = cfg["data"]["prob_col"],
        home_fav_col = cfg["data"]["home_fav_col"],
        fav_odds_col = cfg["data"]["fav_odds_col"],
        dog_odds_col = cfg["data"]["dog_odds_col"],
        stake_strategy = cfg["stake"]["strategy"],
        flat_stake     = cfg["stake"].get("flat_size", 100.0),
        bankroll       = cfg["stake"].get("bankroll", 1_000.0),
    )

    sim = MonteCarloSimulator(bet_recs)
    summary = sim.run(
        num_sim  = cfg["simulation"].get("num_runs", 100_000),
        plot_path = Path(cfg["simulation"].get("plot_path", "mc_plot.png"))
    )

    print("Expected Return:", f"${summary.expected_return:.2f}")
    print("Std PNL:", f"${summary.std_pnl:.2f}")
    print("T-statistic:", f"{summary.t_stat:.2f}")
    print("P(loss):", f"{summary.prob_loss:.2%}")
    print("Range:", f"${summary.min_pnl:.2f} — ${summary.max_pnl:.2f}")
    print("25th/75th percentiles:", f"${summary.pct25:.2f} / ${summary.pct75:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte-Carlo capital allocation evaluation")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
