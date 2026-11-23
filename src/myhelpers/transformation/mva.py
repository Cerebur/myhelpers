"""Minimum variance analysis and event parameter searches."""

from typing import Any, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from myhelpers.transformation.pyrf_local import filt, mva as pyrf_mva


def mva(v: Any, cdir: np.ndarray | None = None, irf: bool = True) -> tuple[Any, dict[str, Any]]:
    """Perform minimum variance analysis on a three-component time series.

    Args:
        v: Input vector series with three components per sample.
        cdir: Optional projection direction; projection is currently unsupported.
        irf: Use the local pyrfu-compatible implementation when true.

    Returns:
        The vector series transformed into the eigenvector basis and a mapping
        containing eigenvalues, eigenvectors, variance ratios, and quality.
    """
    # --- Covariance matrix ---
    v_cov = np.cov(v, rowvar=False, bias=True)

    ## --- Optional projection onto plane orthogonal to cdir ---
    if cdir is not None:
        raise NotImplementedError("cdir is not implemented")
    #    ccol = cdir.reshape(3, 1)
    #    cunit = ccol / np.linalg.norm(ccol)
    #    D = np.eye(3) - np.dot(cunit, cunit.T)
    #    v_cov = D @ v_cov @ D

    if not irf:
        # --- Eigen decomposition ---
        eigvals, eigvecs = np.linalg.eigh(v_cov)

        # eigvals ascending → eigvecs[:,0] = minimum variance

        if (
            eigvecs[:, 0][0] < 0
        ):  # this ensures that the x component of the minimum variance is always positive
            eigvecs[:, 0] = -eigvecs[:, 0]
            eigvecs[:, 1] = -eigvecs[:, 1]  # to keep a right handed system

        λ3, λ2, λ1 = eigvals  # rename for clarity
    else:
        _, eigvals, eigvecs = pyrf_mva(v)
        # Reverse eigvals to ascending order
        eigvals = eigvals[::-1]
        λ3, λ2, λ1 = eigvals
        # Likewise, reorder columns
        eigvecs = eigvecs[:, [2, 1, 0]]

        # Enforce the same sign convention as irf=False
        if eigvecs[0, 0] < 0:
            eigvecs[:, 0] = -eigvecs[:, 0]
            eigvecs[:, 1] = -eigvecs[:, 1]
            eigvecs[:, 2] = -eigvecs[:, 2]

    out = v.copy()
    out.data = v.data @ eigvecs

    # --- Quality metrics ---
    ratio12 = λ1 / λ2 if λ2 > 0 else np.inf
    ratio23 = λ2 / λ3 if λ3 > 0 else np.inf

    # Basic quality classification (based on standard MVA criteria)
    if ratio23 > 10:
        quality = "High-quality normal (λ2/λ3 ≫ 10)"
    elif 3 < ratio23 <= 10:
        quality = "Moderate-quality normal"
    elif 1.5 < ratio23 <= 3:
        quality = "Low-quality normal (degenerate eigenvalues)"
    else:
        quality = "Unreliable — λ2 ≈ λ3; normal poorly determined"

    return out, {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "normal": eigvecs[:, 0],
        "intermediate": eigvecs[:, 1],
        "maximum": eigvecs[:, 2],
        "ratio12": ratio12,
        "ratio23": ratio23,
        "quality": quality,
    }


def analyze_events_mva(
    b_full: Any, tint: Sequence[Any], freq_range: Sequence[float],
    time_jitters_sec: Sequence[float]
) -> pd.DataFrame:
    """Evaluate MVA results across filter frequencies and boundary jitters.

    Args:
        b_full: Full magnetic field time series.
        tint: Event start and end times.
        freq_range: Filter cutoff frequencies in hertz.
        time_jitters_sec: Symmetric event-boundary extensions in seconds.

    Returns:
        A DataFrame with one row for each valid frequency and jitter pair.
    """
    p_start, p_end = tint
    t0 = pd.to_datetime(p_start)
    t1 = pd.to_datetime(p_end)
    jitter_runs = []

    for f_cut in tqdm(freq_range, desc="Frequencies"):
        b_filt = filt(b_full, 0, f_cut, 3)
        for dt in time_jitters_sec:
            dt_end = dt_start = dt
            sub_t0 = t0 - pd.Timedelta(seconds=dt_start)
            sub_t1 = t1 + pd.Timedelta(seconds=dt_end)

            if sub_t0 >= sub_t1:
                continue

            b_slice = b_filt.sel(time=slice(sub_t0, sub_t1))
            if len(b_slice) < 4:
                continue

            _, mva_res = mva(b_slice, irf=True)
            # l3, l2, l1 = mva_res["eigvals"]
            ratio23 = mva_res["ratio23"]
            ratio12 = mva_res["ratio12"]

            norm_vec = mva_res["normal"] / np.linalg.norm(mva_res["normal"])

            jitter_runs.append(
                {
                    "freq": f_cut,
                    "t_start": sub_t0,
                    "t_end": sub_t1,
                    "dt_start": dt_start,
                    "dt_end": dt_end,
                    "ratio23": ratio23,
                    "ratio12": ratio12,
                    "normal_vec": norm_vec,
                    "mva_res": mva_res,
                }
            )

    df_results = pd.DataFrame(jitter_runs)

    return df_results


def analyze_events_mva_gemini(
    b_full: Any, event_candidates: Sequence[Sequence[Any]], freq_range: Sequence[float],
    time_jitters_sec: Sequence[float] = (0, -5, 5), angular_threshold_deg: float = 15.0,
    min_lambda3_ratio: float = 1e-4, max_lambda3_ratio: float = 100,
    irf: bool = True, event_prefix: str = "Event"
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    """Search stable filter frequencies and event boundaries for MVA.

    Args:
        b_full: Full magnetic field time series.
        event_candidates: Start/end time pairs for candidate events.
        freq_range: Filter cutoff frequencies in hertz.
        time_jitters_sec: Time shifts applied to both event boundaries.
        angular_threshold_deg: Maximum cross-frequency angular spread for stability.
        min_lambda3_ratio: Maximum fraction of the largest eigenvalue allowed
            for the smallest eigenvalue to be considered physically plausible.
        max_lambda3_ratio: Upper bound for the intermediate-to-minimum variance ratio.
        irf: Use the local pyrfu-compatible MVA implementation when true.
        event_prefix: Prefix used to construct event identifiers.

    Returns:
        A summary DataFrame and per-event DataFrames for base and jitter runs.
    """
    summary_results = []
    all_details = {}

    for idx in range(len(event_candidates)):
        p_start, p_end = event_candidates[idx]
        t0 = pd.to_datetime(p_start)
        t1 = pd.to_datetime(p_end)
        event_id = f"{event_prefix}_{idx + 1}_{t0.strftime('%H%M%S')}"

        # ------------------------------------------------------------------
        # STAGE 1: Frequency Sweep on Base Window (No Jitters)
        # ------------------------------------------------------------------
        base_runs = []
        for f_cut in tqdm(freq_range, desc=f"Stage 1: Frequencies for {event_id}"):
            b_filt = filt(b_full, 0, f_cut, 3)
            b_slice = b_filt.sel(time=slice(t0, t1))

            if len(b_slice) < 4:
                print("Time window too short, skipping this")
                continue

            _, mva_res = mva(b_slice, irf=irf)
            l3, l2, l1 = mva_res["eigvals"]
            raw_ratio23 = mva_res["ratio23"]

            is_physical = (
                np.isfinite(raw_ratio23)
                and (l3 < min_lambda3_ratio * l1)
                and (l3 > 0)
                and (raw_ratio23 < max_lambda3_ratio)
            )

            norm_vec = mva_res["normal"] / np.linalg.norm(mva_res["normal"])

            base_runs.append(
                {
                    "freq": f_cut,
                    "ratio23": raw_ratio23 if is_physical else 0.0,
                    "is_physical": is_physical,
                    "normal_vec": norm_vec,
                }
            )

        df_base: Any = pd.DataFrame(base_runs)
        if df_base.empty:
            continue

        # Calculate cross-frequency angular dispersion
        spreads = []
        for i, row in df_base.iterrows():
            if not row["is_physical"]:
                spreads.append(np.nan)
                continue

            f_val = row["freq"]
            n_val = row["normal_vec"]

            # Compare to neighboring frequencies within ~1 octave
            neighbors = df_base[
                (df_base["freq"] >= 0.5 * f_val)
                & (df_base["freq"] <= 2.0 * f_val)
                & (df_base["is_physical"])
            ]

            if len(neighbors) > 0:
                print("###############")
                print(neighbors["normal_vec"])
                normals_mat = np.vstack(neighbors["normal_vec"].values)
                dots = np.abs(np.clip(np.dot(normals_mat, n_val), -1.0, 1.0))
                spreads.append(np.mean(np.degrees(np.arccos(dots))))
            else:
                spreads.append(np.nan)

        df_base["cross_freq_spread_deg"] = spreads

        # ------------------------------------------------------------------
        # STAGE 2: Identify Target Frequencies
        # ------------------------------------------------------------------
        stable_mask = df_base["cross_freq_spread_deg"] <= angular_threshold_deg

        if stable_mask.any():
            target_freqs = df_base[stable_mask]["freq"].values
        else:
            # Fallback: Pick the one frequency with the lowest angular spread
            best_base_idx = df_base["cross_freq_spread_deg"].idxmin()
            target_freqs = [df_base.loc[best_base_idx, "freq"]]

        # ------------------------------------------------------------------
        # STAGE 3: Time Boundary Jitters on Target Frequencies Only
        # ------------------------------------------------------------------
        jitter_runs = []
        for f_cut in tqdm(target_freqs, desc=f"Stage 2: Jitters for {event_id}"):
            b_filt = filt(b_full, 0, f_cut, 3)

            for dt_start in time_jitters_sec:
                dt_end = dt_start
                sub_t0 = t0 + pd.Timedelta(seconds=dt_start)
                sub_t1 = t1 + pd.Timedelta(seconds=dt_end)

                if sub_t0 >= sub_t1:
                    continue

                b_slice = b_filt.sel(time=slice(sub_t0, sub_t1))
                if len(b_slice) < 4:
                    continue

                _, mva_res = mva(b_slice, irf=irf)
                l3, l2, l1 = mva_res["eigvals"]
                raw_ratio23 = mva_res["ratio23"]

                is_physical = (
                    np.isfinite(raw_ratio23)
                    and (l3 < min_lambda3_ratio * l1)
                    and (l3 > 0)
                    and (raw_ratio23 < max_lambda3_ratio)
                )

                norm_vec = mva_res["normal"] / np.linalg.norm(mva_res["normal"])

                jitter_runs.append(
                    {
                        "freq": f_cut,
                        "t_start": sub_t0,
                        "t_end": sub_t1,
                        "dt_start": dt_start,
                        "dt_end": dt_end,
                        "ratio23": raw_ratio23 if is_physical else 0.0,
                        "raw_ratio23": raw_ratio23,
                        "is_physical": is_physical,
                        "normal_vec": norm_vec,
                        "mva_res": mva_res,
                    }
                )

        df_jitters: Any = pd.DataFrame(jitter_runs)

        # ------------------------------------------------------------------
        # STAGE 4: Select Optimum
        # ------------------------------------------------------------------
        if df_jitters.empty:
            continue

        phys_pool = df_jitters[df_jitters["is_physical"]]
        if not phys_pool.empty:
            best_idx = phys_pool["ratio23"].idxmax()
        else:
            best_idx = df_jitters["ratio23"].idxmax()

        best = df_jitters.loc[best_idx]

        summary_results.append(
            {
                "event_id": event_id,
                "orig_start": t0,
                "orig_end": t1,
                "opt_freq_hz": best["freq"],
                "opt_start": best["t_start"],
                "opt_end": best["t_end"],
                "max_ratio23": best["ratio23"],
                "raw_ratio23": best["raw_ratio23"],
                "vector_stable": stable_mask.any(),
                "opt_normal": best["normal_vec"],
                "opt_mva_res": best["mva_res"],
            }
        )

        # Save both dataframes for the plotting function
        all_details[event_id] = {"base": df_base, "jitters": df_jitters}

    summary_df = pd.DataFrame(summary_results)
    return summary_df, all_details

