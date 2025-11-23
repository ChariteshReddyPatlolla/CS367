import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# HMM library
from hmmlearn.hmm import GaussianHMM


import yfinance as yf

sns.set(style="whitegrid", context="talk")


def download_data(ticker="^GSPC", start="2015-01-01", end=None):
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError(f"No data downloaded for {ticker} between {start} and {end}.")
    df = df[["Adj Close"]].rename(columns={"Adj Close": "adj_close"})
    df.index = pd.to_datetime(df.index)
    return df


def compute_returns(df, kind="log"):
    df = df.copy()
    if kind == "log":
        df["return"] = np.log(df["adj_close"]).diff()
    else:
        df["return"] = df["adj_close"].pct_change()
    df = df.dropna(subset=["return"])

    return df

#-------- MODEL TRAINING -------------

def fit_hmm(returns, n_states=3, n_init=10, cov_type="diag", random_state=42, max_iter=200):
    """
    Fit Gaussian HMM using multiple random restarts and return the best model (highest log-likelihood).
    returns: 1-D numpy array of shape (T,)
    """
    X = returns.reshape(-1, 1)
    best_score = -np.inf
    best_model = None

    for seed in range(random_state, random_state + n_init):
        model = GaussianHMM(n_components=n_states,
                            covariance_type=cov_type,
                            n_iter=max_iter,
                            random_state=seed,
                            verbose=False)
        try:
            model.fit(X)
            score = model.score(X)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception as e:
            print(f"[init seed={seed}] fit failed: {e}")

    if best_model is None:
        raise RuntimeError("HMM fitting failed for all initializations.")
    return best_model, best_score


def decode_states(model, returns):
    X = returns.reshape(-1, 1)
    states = model.predict(X)
    posteriors = model.predict_proba(X)
    return states, posteriors

#----model paramters extraction-------
def state_stats(model):
    means = np.squeeze(model.means_)  # shape (n_states, 1) -> squeeze -> (n_states,)
    # handle covariance shapes for different covariance_type values
    if model.covariance_type == "diag":
        vars_ = np.squeeze(model.covars_)  # (n_states, 1) -> (n_states,)
    elif model.covariance_type == "full":
        # full covariance: covars_ is (n_states, 1, 1) for 1-dim observations
        vars_ = np.array([cov[0, 0] for cov in model.covars_])
    else:
        # tied or spherical: covars_ might be scalar or (n_states,)
        vars_ = np.squeeze(model.covars_)

    transmat = model.transmat_
    startprob = model.startprob_
    return means, np.sqrt(vars_), vars_, transmat, startprob


def save_csv(df, states, outpath):
    out = df.copy()
    out["state"] = states
    out.to_csv(outpath, index=True)
    print(f"Saved decoded states CSV to: {outpath}")

#--------Visulaization---------

def plot_price_states(df, states, outpath):
    plt.figure(figsize=(14, 6))
    palette = sns.color_palette("tab10", n_colors=len(np.unique(states)))
    for s in np.unique(states):
        mask = (states == s)
        plt.plot(df.index[mask], df["adj_close"].values[mask], '.', label=f"State {s}", alpha=0.7)
    plt.plot(df.index, df["adj_close"].values, color="0.6", alpha=0.2, linewidth=0.8)  # full price as faint line
    plt.legend()
    plt.title("Adjusted Close Price colored by inferred HMM state")
    plt.xlabel("Date")
    plt.ylabel("Adj Close")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved price-by-state plot to: {outpath}")


def plot_returns_states(df, states, outpath):
    plt.figure(figsize=(14, 6))
    palette = sns.color_palette("tab10", n_colors=len(np.unique(states)))
    for s in np.unique(states):
        mask = (states == s)
        plt.scatter(df.index[mask], df["return"].values[mask], s=8, label=f"State {s}", alpha=0.6)
    plt.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    plt.legend()
    plt.title("Daily returns colored by inferred HMM state")
    plt.xlabel("Date")
    plt.ylabel("Return (log)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved returns-by-state plot to: {outpath}")


def plot_transition_heatmap(transmat, outpath):
    plt.figure(figsize=(6, 5))
    sns.heatmap(transmat, annot=True, fmt=".3f", cmap="viridis", cbar=True,
                xticklabels=[f"S{i}" for i in range(1, transmat.shape[0] + 1)],
                yticklabels=[f"S{i}" for i in range(1, transmat.shape[0] + 1)])
    plt.title("HMM Transition Matrix")
    plt.xlabel("To state")
    plt.ylabel("From state")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"Saved transition heatmap to: {outpath}")


def print(means, stds, vars_, transmat, startprob):
    n = len(means)
    print("\n HMM Estimated Parameters ")
    for i in range(n):
        print(f"State {i}: mean = {means[i]:.6f}, std = {stds[i]:.6f}, var = {vars_[i]:.6e}")
    print("\nTransition matrix:")
    print(np.round(transmat, 4))
    print("\nStart probabilities:")
    print(np.round(startprob, 4))


def main(args):
    ticker = args.ticker
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print(f"Downloading {ticker} from {args.start} to {args.end or 'today'} ")
    df = download_data(ticker=ticker, start=args.start, end=args.end)

    df = compute_returns(df, kind="log")
    returns = df["return"].values

    print("Fitting HMM ")
    model, score = fit_hmm(returns, n_states=args.n_states, n_init=args.n_init,
                           cov_type=args.cov_type, random_state=args.seed, max_iter=args.max_iter)
    print(f"Best model log-likelihood: {score:.2f}")

    states, posteriors = decode_states(model, returns)
    means, stds, vars_, transmat, startprob = state_stats(model)

    summarize_and_print(means, stds, vars_, transmat, startprob)

    # Clean ticker for filenames (remove ^ and other problematic chars)
    ticker_clean = ticker.replace("^", "").replace("/", "_")

    # Save CSV
    csv_path = os.path.join(outdir, f"{ticker_clean}_hmm_states.csv")
    save_csv(df, states, csv_path)

    # Plots
    plot_price_states(df, states, os.path.join(outdir, f"{ticker_clean}_states_price.png"))
    plot_returns_states(df, states, os.path.join(outdir, f"{ticker_clean}_returns_states.png"))
    plot_transition_heatmap(transmat, os.path.join(outdir, f"{ticker_clean}_transition_heatmap.png"))

    print("\nDone.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=" Gaussian HMM regime analysis on stock market returns."
    )

    # Data and output
    data = parser.add_argument_group("Data settings")
    data.add_argument("--ticker", default="^GSPC", help="Ticker symbol")
    data.add_argument("--start", default="2015-01-01", help="Start date")
    data.add_argument("--end", default=None, help="End date")
    data.add_argument("--outdir", default="output", help="Where to save results")


    # Model parameters
    model = parser.add_argument_group("Model settings")
    model.add_argument("--n-states", type=int, default=3, help="HMM states")
    model.add_argument("--n-init", type=int, default=10, help="Random restarts")
    model.add_argument("--cov-type", default="diag",
                       choices=["full", "diag", "tied", "spherical"])
    model.add_argument("--max-iter", type=int, default=200)
    model.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)
