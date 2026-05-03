#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_recompute.py
===================
Standalone re-loads Allantis MT + PUT_SKEW daily and prints ALL the numbers
needed to populate the hardcoded sections of the PUT_SKEW_NIVEL_ALLANTIS dashboard.

Independent of generate_evidence.py: re-loads data with different code path
(no shared imports, intentional duplication for audit purposes).

Run:
    python verify_recompute.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ALLANTIS_CSV = Path(
    r"C:/Users/Administrator/Desktop/BULK OPTIONSTRAT/ESTRATEGIAS/Allantis/LIVE"
    r"/[MAIN RANKEO MT]_combined_ALLANTIS_ALLDAYS.csv"
)
SKEW_CSV = Path(
    r"C:/Users/Administrator/Desktop/BULK OPTIONSTRAT/ESTRATEGIAS/Skew/SKEW_PUT_ENRICHED.csv"
)

REGIME_FAV = 80.0
REGIME_ADV = 20.0
REF_HORIZON = 30
SPX_THR = 3.0
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 42


def stats(s):
    s = pd.to_numeric(s, errors="coerce").dropna()
    if s.empty:
        return dict(N=0, mean=np.nan, median=np.nan, WR=np.nan, PF=np.nan)
    gw = s[s > 0].sum(); gl = -s[s < 0].sum()
    return dict(
        N=len(s),
        mean=float(s.mean()),
        median=float(s.median()),
        WR=100.0 * float((s > 0).mean()),
        PF=gw/gl if gl > 0 else np.nan,
    )


def main():
    print("=" * 100)
    print("PUT_SKEW_NIVEL_ALLANTIS — verify_recompute (cross-check vs generate_evidence.py)")
    print("=" * 100)

    # Load Allantis (BOM UTF-8)
    df = pd.read_csv(ALLANTIS_CSV, encoding="utf-8-sig", low_memory=False)
    df.columns = [c.replace("﻿", "") for c in df.columns]
    df = df.rename(columns={"dia": "trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["trade_date"])
    print(f"\n[1] Allantis loaded: {len(df):,} rows ({df['trade_date'].min().date()} -> {df['trade_date'].max().date()})")

    # Load PUT_SKEW daily
    s = pd.read_csv(SKEW_CSV, low_memory=False)
    s = s[(s["snapshot_time"] == "10:30:00") & (s["dte_target"] == 60) & (s["side"] == "PUT")].copy()
    s["trade_date"] = pd.to_datetime(s["trade_date"], errors="coerce").dt.normalize()
    s = s.dropna(subset=["trade_date", "skew_25d_vs50_pct_expanding"]).drop_duplicates("trade_date")
    s = s.rename(columns={"skew_25d_vs50_pct_expanding": "PUT_SKEW_PCT"})
    print(f"[2] PUT_SKEW daily: {len(s):,} rows")

    # Join
    n_before = len(df)
    df = df.merge(s[["trade_date", "PUT_SKEW_PCT"]], on="trade_date", how="inner")
    df = df.dropna(subset=["PUT_SKEW_PCT"])
    n_unfiltered = len(df)
    print(f"[3] Joined: {n_unfiltered:,}/{n_before:,} have PUT_SKEW ({100*n_unfiltered/n_before:.1f}%)")

    # Sanity check SPX filter
    spx = pd.to_numeric(df["SPX_chg_pct_d030"], errors="coerce")
    print(f"[4] SPX_chg_pct_d030 stats:  min={spx.min():.2f}  max={spx.max():.2f}  std={spx.std():.2f}  (units: pp expected)")

    # Apply filter
    mask = spx.abs() <= SPX_THR
    df_full = df.copy()
    df = df[mask].copy()
    n_filt = len(df)
    print(f"[5] Filtered |SPX|<={SPX_THR}%: {n_filt:,}/{n_unfiltered:,} retained ({100*n_filt/n_unfiltered:.1f}%)")

    # Baseline universe
    print("\n" + "=" * 100)
    print("BASELINE UNIVERSO (Allantis filtered |SPX|<=3%)")
    print("=" * 100)
    pnl_ref = f"PnL_d{REF_HORIZON:03d}_mediana"
    pnl_50 = "PnL_d050_mediana"
    s30 = stats(df[pnl_ref])
    s50 = stats(df[pnl_50])
    y_min = df["trade_date"].min().year
    y_max = df["trade_date"].max().year
    print(f"\n[BASELINE]  N={s30['N']:,}  range={y_min}-{y_max}")
    print(f"  d030: WR={s30['WR']:.1f}%  mean={s30['mean']:+.2f}  PF={s30['PF']:.2f}")
    print(f"  d050: WR={s50['WR']:.1f}%  mean={s50['mean']:+.2f}  PF={s50['PF']:.2f}")
    print(f"\n[HTML rules-baseline line]")
    print(f"  Baseline universo Allantis MT ({s30['N']:,} trades, {y_min}-{y_max}, |SPX|<=3%):")
    print(f"  WR {s30['WR']:.0f}% / {s50['WR']:.0f}%  *  mean +{s30['mean']:.0f} / +{s50['mean']:.0f} pts  *  PF {s30['PF']:.1f} / {s50['PF']:.1f}")
    print(f"  (d030 / d050)")

    # Regime split
    print("\n" + "=" * 100)
    print("REGIME SPLIT — FAV/NEU/ADV (filtered)")
    print("=" * 100)
    regime_results = {}
    for label, mask_reg in [
        ("FAVORABLE", df["PUT_SKEW_PCT"] >= REGIME_FAV),
        ("NEUTRAL",   (df["PUT_SKEW_PCT"] > REGIME_ADV) & (df["PUT_SKEW_PCT"] < REGIME_FAV)),
        ("ADVERSO",   df["PUT_SKEW_PCT"] <= REGIME_ADV),
    ]:
        sub = df[mask_reg]
        s30r = stats(sub[pnl_ref])
        s50r = stats(sub[pnl_50])
        pct_univ = 100.0 * len(sub) / s30["N"]
        regime_results[label] = (s30r, s50r, pct_univ)
        print(f"\n[{label}] N={len(sub):,} ({pct_univ:.0f}% del universo)")
        print(f"  d030: WR={s30r['WR']:.1f}%  mean={s30r['mean']:+.2f}  PF={s30r['PF']:.2f}")
        print(f"  d050: WR={s50r['WR']:.1f}%  mean={s50r['mean']:+.2f}  PF={s50r['PF']:.2f}")

    print("\n[HTML rules-table cells (vs baseline universo)]")
    print(f"  {'Banda':<15} {'WR vs univ':<20} {'mean vs univ':<20} {'PF vs univ':<20} {'% univ':<10}")
    for label in ["FAVORABLE", "NEUTRAL", "ADVERSO"]:
        s30r, s50r, pct = regime_results[label]
        wr_d30 = s30r['WR'] - s30['WR']
        wr_d50 = s50r['WR'] - s50['WR']
        m_d30 = s30r['mean'] / s30['mean'] if s30['mean'] != 0 else np.nan
        m_d50 = s50r['mean'] / s50['mean'] if s50['mean'] != 0 else np.nan
        pf_d30 = s30r['PF'] / s30['PF'] if (np.isfinite(s30['PF']) and s30['PF'] > 0) else np.nan
        pf_d50 = s50r['PF'] / s50['PF'] if (np.isfinite(s50['PF']) and s50['PF'] > 0) else np.nan
        print(f"  {label:<15} {wr_d30:+.0f}pp / {wr_d50:+.0f}pp     "
              f"{m_d30:.1f}x / {m_d50:.1f}x       "
              f"{pf_d30:.1f}x / {pf_d50:.1f}x       "
              f"{pct:.0f}%")

    # Conditional during-trade (PUT_SKEW asof lookup)
    print("\n" + "=" * 100)
    print("CONDITIONAL DURING-TRADE  (entry FAV)")
    print("=" * 100)
    fav = df[df["PUT_SKEW_PCT"] >= REGIME_FAV].copy()
    print(f"FAV entries (filtered): N={len(fav):,}")

    s_full = pd.read_csv(SKEW_CSV, usecols=["trade_date","snapshot_time","dte_target","side","skew_25d_vs50_pct_expanding"], low_memory=False)
    s_full = s_full[(s_full["snapshot_time"] == "10:30:00") & (s_full["dte_target"] == 60) & (s_full["side"] == "PUT")]
    s_full["trade_date"] = pd.to_datetime(s_full["trade_date"], errors="coerce").dt.normalize()
    s_full["PUT_SKEW_PCT"] = pd.to_numeric(s_full["skew_25d_vs50_pct_expanding"], errors="coerce")
    s_full = s_full.dropna(subset=["trade_date","PUT_SKEW_PCT"]).drop_duplicates("trade_date").sort_values("trade_date")
    ts_dates = s_full["trade_date"].to_numpy(dtype="datetime64[ns]")
    ts_vals = s_full["PUT_SKEW_PCT"].to_numpy(dtype=float)

    def asof(date):
        if pd.isna(date):
            return np.nan
        # Convert to datetime64[ns] to match ts_dates dtype
        d64 = np.datetime64(pd.Timestamp(date), "ns")
        pos = np.searchsorted(ts_dates, d64, side="right") - 1
        if pos < 0:
            return np.nan
        return float(ts_vals[pos])

    if len(fav) > 0:
        future_dates = fav["trade_date"].to_numpy(dtype="datetime64[ns]") + np.timedelta64(REF_HORIZON, "D")
        ten_at_30 = np.array([asof(pd.Timestamp(d)) for d in future_dates])
        still_high = ten_at_30 >= REGIME_FAV
        pct_remain = round(100 * np.mean(still_high))
        n_remain = int(np.sum(still_high))
        sub_remain = fav.iloc[still_high]
        srem = stats(sub_remain[pnl_50])
        ratio_mean = srem['mean'] / s50['mean'] if s50['mean'] != 0 else np.nan
        print(f"\n[A] Of FAV entries, {n_remain:,}/{len(fav):,} ({pct_remain}%) have PUT_SKEW>=80 still at entry+{REF_HORIZON}d")
        print(f"    d050 conditional: WR={srem['WR']:.0f}%  mean={srem['mean']:+.0f}pts ({ratio_mean:.1f}x universo)  PF={srem['PF']:.0f}")

        # % drop ≤20
        any_drop = np.zeros(len(fav), dtype=bool)
        for dt_offset in [10, 20, 30, 40, 50]:
            future_dates = fav["trade_date"].to_numpy(dtype="datetime64[ns]") + np.timedelta64(dt_offset, "D")
            ten_x = np.array([asof(pd.Timestamp(d)) for d in future_dates])
            any_drop |= (ten_x <= REGIME_ADV)
        pct_drop = round(100 * np.mean(any_drop))
        n_drop = int(np.sum(any_drop))
        sub_drop = fav.iloc[any_drop]
        sdrop = stats(sub_drop[pnl_50])
        print(f"\n[B] Of FAV entries, {n_drop:,}/{len(fav):,} ({pct_drop}%) see PUT_SKEW<=20 algun dia en proximos 50d")
        print(f"    d050 conditional: N={sdrop['N']:,}  mean={sdrop['mean']:+.2f}")

    # Year stability
    print("\n" + "=" * 100)
    print("YEAR STABILITY 2019-2025  (FAV cohort vs baseline, filtered)")
    print("=" * 100)
    df["year"] = df["trade_date"].dt.year
    print(f"\n  {'Year':<6} {'N_fav':>7} {'mean_fav':>10} {'WR_fav':>8} {'baseline':>10} {'WR_base':>8}  {'verdict':<14}")
    n_beats = 0
    fav_underperform = []
    fav_zero = []
    for y, g in df.groupby("year"):
        fav_g = g[g["PUT_SKEW_PCT"] >= REGIME_FAV]
        s_fav = stats(fav_g[pnl_ref])
        s_base = stats(g[pnl_ref])
        if s_fav['N'] == 0:
            verdict = "SIN FAV"
            fav_zero.append(y)
        elif s_fav['mean'] < s_base['mean']:
            verdict = "FAV PEOR"
            fav_underperform.append((y, s_fav['mean'], s_base['mean']))
        else:
            verdict = "FAV bate"
            n_beats += 1
        print(f"  {y:<6} {s_fav['N']:>7} {s_fav['mean']:>+10.2f} {s_fav['WR']:>7.1f}% {s_base['mean']:>+10.2f} {s_base['WR']:>7.1f}%  {verdict}")

    n_years = df['year'].nunique()
    print(f"\n[HTML caveat line]: FAV bate al universo en {n_beats} de {n_years} anios.")
    if fav_underperform:
        details = ", ".join([f"{y} (FAV mean {m:+.0f})" for y, m, b in fav_underperform])
        print(f"  Fallos: {details}")
    if fav_zero:
        details = ", ".join([f"{y} (sin FAV)" for y in fav_zero])
        print(f"  Sin FAV: {details}")

    # Headline Spearman
    print("\n" + "=" * 100)
    print(f"HEADLINE SPEARMAN d{REF_HORIZON:03d} (filtered |SPX|<=3%)")
    print("=" * 100)
    sub = df[["PUT_SKEW_PCT", pnl_ref]].dropna()
    r = spearmanr(sub["PUT_SKEW_PCT"], sub[pnl_ref]).correlation
    score_rank = pd.Series(sub["PUT_SKEW_PCT"].values).rank().to_numpy(dtype=float)
    pnl_rank = pd.Series(sub[pnl_ref].values).rank().to_numpy(dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + REF_HORIZON)
    sp_vals = np.full(BOOTSTRAP_N, np.nan, dtype=float)
    n_pts = len(sub)
    for b in range(BOOTSTRAP_N):
        idx = rng.integers(0, n_pts, size=n_pts)
        sr = score_rank[idx]; pr = pnl_rank[idx]
        sx = sr.std(); sy = pr.std()
        if sx > 0 and sy > 0:
            sp_vals[b] = float(np.mean((sr - sr.mean()) * (pr - pr.mean())) / (sx * sy))
    ci_lo = float(np.nanpercentile(sp_vals, 2.5))
    ci_hi = float(np.nanpercentile(sp_vals, 97.5))
    print(f"  N={n_pts:,}  Spearman r = {r:+.4f}  CI95% = [{ci_lo:+.4f}, {ci_hi:+.4f}]")

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)


if __name__ == "__main__":
    main()
