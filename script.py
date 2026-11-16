import pandas as pd
import numpy as np

def main():
    # 1. Load datasets
    bbb = pd.read_csv("ball_by_ball.csv", low_memory=False)
    matches = pd.read_csv("matches.csv", low_memory=False)
    comps = pd.read_csv("competitions.csv", low_memory=False)
    ctpm = pd.read_csv("competition_team_player_mapping.csv", low_memory=False)

    # 2. Join match & competition metadata (2022+, regular innings 1 & 2, IPL & T20I)
    df = bbb.merge(matches, on="match_id", how="inner").merge(comps, on="comp_id", how="inner")
    df["year"] = pd.to_datetime(df["match_date"], errors="coerce").dt.year
    df["comp_tier"] = np.where(
        df["comp_name"].str.contains("IPL", case=False, na=False),
        "IPL",
        np.where(df["match_format"] == "T20I", "T20I", "Other_T20"),
    )
    df = df[(df["year"] >= 2022) & (df["innings"].isin([1, 2])) & (df["comp_tier"].isin(["IPL", "T20I"]))].copy()
    df = df.sort_values(["match_id", "innings", "overid", "over_ball_num", "innings_ball_num"]).reset_index(drop=True)

    # 3. Exact ball, run, and wicket accounting
    df["extras_clean"] = df["extras_type"].fillna("none")
    df["is_wide"] = (df["extras_clean"] == "wide").astype(int)
    df["is_noball"] = df["extras_clean"].isin(["Overstep", "other"]).astype(int)
    df["is_legal_ball"] = ((df["is_wide"] == 0) & (df["is_noball"] == 0)).astype(int)
    df["is_batter_ball"] = (df["is_wide"] == 0).astype(int)

    df["total_runs"] = df["runs"] + df["extras1_runs"] + df["extras_2_runs"]
    df["batter_runs"] = df["runs"]
    df["bowler_runs"] = df["runs"] + np.where(
        df["extras_clean"].isin(["wide", "Overstep", "other"]), df["extras1_runs"], 0
    )

    non_bowler_wkts = {"RunOut", "Retire Out", "Obstructing the field", "Obstracting the field"}
    df["is_bowler_wkt"] = (df["is_wicket"] & (~df["dismissal_type"].isin(non_bowler_wkts))).astype(int)
    df["is_striker_out"] = (df["is_wicket"] & (df["dismissed_player_id"] == df["striker_id"])).astype(int)
    df["is_team_wkt"] = df["is_wicket"].astype(int)
    df["is_boundary"] = (df["is_four"] | df["is_six"]).astype(int)

    # 4. Pre-delivery cumulative state & DLS-aware targets
    grp = df.groupby(["match_id", "innings"])
    df["cum_runs"] = grp["total_runs"].cumsum() - df["total_runs"]
    df["cum_wkts"] = grp["is_team_wkt"].cumsum() - df["is_team_wkt"]
    df["legal_balls_bowled"] = grp["is_legal_ball"].cumsum() - df["is_legal_ball"]

    df["inn_max_balls"] = np.where(
        df["innings"] == 1,
        df["first_inn_revised_overs"].fillna(120.0),
        df["second_inn_revised_overs"].fillna(120.0),
    )
    df["balls_remaining"] = np.clip(df["inn_max_balls"] - df["legal_balls_bowled"], 1.0, 120.0)

    inn1_totals = df[df["innings"] == 1].groupby("match_id")["total_runs"].sum().rename("inn1_total")
    df = df.merge(inn1_totals, on="match_id", how="left")
    df["target"] = np.where(df["innings"] == 2, df["revised_target"].fillna(df["inn1_total"] + 1.0), np.nan)
    df["runs_needed"] = np.where(df["innings"] == 2, np.clip(df["target"] - df["cum_runs"], 0.0, None), np.nan)
    df["rrr"] = np.where(df["innings"] == 2, (df["runs_needed"] * 6.0) / df["balls_remaining"], np.nan)
    df["crr"] = np.where(df["legal_balls_bowled"] > 0, (df["cum_runs"] * 6.0) / df["legal_balls_bowled"], 8.0)

    # 5. Delivery-level Pressure Index (PI in [0, 1])
    comp_par_rpo = df.groupby(["comp_tier", "year"])["total_runs"].transform("mean") * 6.0
    rrr_ratio = df["rrr"] / comp_par_rpo
    p_rrr = 1.0 / (1.0 + np.exp(-3.2 * (rrr_ratio - 1.22)))
    p_wkts = (np.clip(df["cum_wkts"], 0, 10) / 10.0) ** 1.3
    p_endgame = np.clip((60.0 - df["balls_remaining"]) / 60.0, 0.0, 1.0)
    garbage_mask = ((df["rrr"] > 24.0) & (df["cum_wkts"] >= 8)) | ((df["rrr"] < 4.5) & (df["cum_wkts"] <= 3))
    pi_inn2 = np.where(garbage_mask, 0.25, 0.45 * p_rrr + 0.30 * p_wkts + 0.25 * p_endgame)

    frac_completed = df["legal_balls_bowled"] / df["inn_max_balls"]
    p_collapse = np.where(
        (frac_completed <= 0.30) & (df["cum_wkts"] >= 2),
        0.75,
        np.where((frac_completed <= 0.50) & (df["cum_wkts"] >= 3), 0.72, 0.20),
    )
    p_death = np.where(frac_completed >= 0.75, 0.70 + 0.03 * df["cum_wkts"], 0.25)
    p_deficit = np.clip((comp_par_rpo - df["crr"]) / 4.0, 0.0, 1.0) * 0.50
    pi_inn1 = np.maximum.reduce([p_collapse, p_death, p_deficit])

    df["pressure_index"] = np.where(df["innings"] == 2, pi_inn2, pi_inn1)
    df["is_high_pressure"] = df["pressure_index"] >= 0.65

    # 6. Validation split export
    val_split = (
        df.groupby("is_high_pressure")
        .agg(
            deliveries=("total_runs", "count"),
            pct_deliveries=("total_runs", lambda s: round(100.0 * len(s) / len(df), 2)),
            avg_runs_per_ball=("total_runs", lambda s: round(s.mean(), 3)),
            implied_run_rate=("total_runs", lambda s: round(s.mean() * 6.0, 2)),
            boundary_pct=("is_boundary", lambda s: round(100.0 * s.mean(), 2)),
            wicket_rate_pct=("is_team_wkt", lambda s: round(100.0 * s.mean(), 2)),
        )
        .reset_index()
    )
    val_split["situation"] = np.where(val_split["is_high_pressure"], "High Pressure (PI >= 0.65)", "Ordinary (PI < 0.65)")
    val_split.to_csv("output_validation.csv", index=False)

    # 7. State-expected baselines (format, year, phase, pressure)
    df["phase"] = pd.cut(frac_completed, bins=[-0.01, 0.30, 0.75, 1.05], labels=["PP", "Middle", "Death"])
    state_cols = ["comp_tier", "year", "phase", "is_high_pressure"]
    df["exp_batter_runs"] = df.groupby(state_cols, observed=False)["batter_runs"].transform("mean")
    df["exp_striker_out"] = df.groupby(state_cols, observed=False)["is_striker_out"].transform("mean")
    df["exp_bowler_runs"] = df.groupby(state_cols, observed=False)["bowler_runs"].transform("mean")
    df["exp_bowler_wkt"] = df.groupby(state_cols, observed=False)["is_bowler_wkt"].transform("mean")

    player_names = (
        pd.concat(
            [
                ctpm[["player_id", "player_name"]],
                bbb[["striker_id", "striker_name"]].rename(columns={"striker_id": "player_id", "striker_name": "player_name"}),
                bbb[["bowler_id", "bowler_name"]].rename(columns={"bowler_id": "player_id", "bowler_name": "player_name"}),
            ]
        )
        .dropna()
        .drop_duplicates("player_id")
        .set_index("player_id")["player_name"]
    )

    LAMBDA_BAT, LAMBDA_BOWL, K_PRIOR = 12.0, 11.0, 120.0

    def rank_players(is_batter=True, min_hp_balls=150, min_ord_balls=150, min_hp_matches=10):
        if is_batter:
            sub = df[df["is_batter_ball"] == 1].copy()
            role_col = "striker_id"
            ball_col = "is_batter_ball"
            sub["val"] = (sub["batter_runs"] - sub["exp_batter_runs"]) - LAMBDA_BAT * (
                sub["is_striker_out"] - sub["exp_striker_out"]
            )
            scale = 100.0
        else:
            sub = df.copy()
            role_col = "bowler_id"
            ball_col = "is_legal_ball"
            sub["val"] = (sub["exp_bowler_runs"] - sub["bowler_runs"]) + LAMBDA_BOWL * (
                sub["is_bowler_wkt"] - sub["exp_bowler_wkt"]
            )
            scale = 24.0

        agg = (
            sub.groupby([role_col, "is_high_pressure"])
            .agg(balls=(ball_col, "sum"), total_val=("val", "sum"), matches=("match_id", "nunique"))
            .unstack(fill_value=0)
        )
        fmt_hp = (
            sub[sub["is_high_pressure"]]
            .groupby([role_col, "comp_tier"])[ball_col]
            .sum()
            .unstack(fill_value=0)
        )

        hp_balls = agg[("balls", True)]
        ord_balls = agg[("balls", False)]
        hp_matches = agg[("matches", True)]

        valid = (hp_balls >= min_hp_balls) & (ord_balls >= min_ord_balls) & (hp_matches >= min_hp_matches)
        hp_rate = scale * agg[("total_val", True)] / hp_balls.replace(0, np.nan)
        ord_rate = scale * agg[("total_val", False)] / ord_balls.replace(0, np.nan)
        delta = hp_rate - ord_rate.clip(lower=-15.0)

        raw_clutch = 0.70 * hp_rate + 0.30 * delta
        league_mean = raw_clutch[valid].mean()
        shrunk = (hp_balls / (hp_balls + K_PRIOR)) * raw_clutch + (K_PRIOR / (hp_balls + K_PRIOR)) * league_mean

        match_hp = sub[sub["is_high_pressure"]].groupby([role_col, "match_id"])["val"].sum()
        pos_rate = (match_hp > 0).groupby(level=0).mean()

        res = pd.DataFrame(
            {
                "player_id": hp_balls.index,
                "player_name": hp_balls.index.map(player_names),
                "hp_balls": hp_balls.astype(int),
                "ord_balls": ord_balls.astype(int),
                "hp_matches": hp_matches.astype(int),
                "ipl_hp_balls": fmt_hp.reindex(hp_balls.index, fill_value=0).get("IPL", 0).astype(int),
                "t20i_hp_balls": fmt_hp.reindex(hp_balls.index, fill_value=0).get("T20I", 0).astype(int),
                "hp_impact": hp_rate.round(2),
                "ord_baseline": ord_rate.round(2),
                "clutch_delta": delta.round(2),
                "pos_match_rate": (pos_rate * 100.0).round(2),
                "final_rating": (shrunk * (0.80 + 0.20 * pos_rate.fillna(0.5))).round(2),
            }
        )[valid].sort_values("final_rating", ascending=False)

        res.insert(0, "rank", range(1, len(res) + 1))
        return res

    batters_df = rank_players(is_batter=True)
    bowlers_df = rank_players(is_batter=False)

    batters_df.to_csv("output_batters.csv", index=False)
    bowlers_df.to_csv("output_bowlers.csv", index=False)

    print("\n=== PRESSURE VALIDATION SPLIT ===")
    print(val_split[["situation", "deliveries", "pct_deliveries", "implied_run_rate", "boundary_pct", "wicket_rate_pct"]].to_string(index=False))

    print("\n=== TOP 10 BATTERS UNDER PRESSURE ===")
    print(batters_df.head(10).to_string(index=False))

    print("\n=== TOP 10 BOWLERS UNDER PRESSURE ===")
    print(bowlers_df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()