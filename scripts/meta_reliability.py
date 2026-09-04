import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from bootstrap import (
    build_team_match_draw_plan,
    build_team_match_draw_plan_for_blocks,
    draw_multiplicities,
    sample_player_set_rows,
)
from metrics import (
    apply_reliability_attempt_eligibility,
    calculate_tier1_metrics,
)
from mlv_adapter import SUPPORTED_MLV_SEASONS, build_mlv_canonical_season
from metric_registry import CONVENTIONAL_METRICS

def require_unique_rows(df, keys, label):
    """Raise a clear error when a statistical unit appears more than once."""
    duplicate_mask = df.duplicated(keys, keep=False)
    if duplicate_mask.any():
        duplicate_keys = (df.loc[duplicate_mask, keys]
                          .drop_duplicates()
                          .head(5)
                          .to_dict('records'))
        raise ValueError(
            f"{label} must be unique on {keys}; duplicate keys include {duplicate_keys}"
        )


class MetaReliabilityEngine:
    """
    A statistical engine designed to evaluate the Stability and Discrimination
    of volleyball metrics through game-level bootstrap resampling, following the
    estimators published in Franks et al. (2016), JQAS 12(4):151-165, Section 3.
    """
    def __init__(self, target_metrics, min_sets=20, seed=42):
        self.target_metrics = target_metrics
        self.min_sets = min_sets
        self.seed = seed

        # Stability CIs use their own deterministic stream. Team-match draws
        # are supplied by the shared bootstrap-plan module.
        self._ci_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 1])
        )

    def _filter_qualifiers(self, df, min_sets=None):
        """Removes low-sample player-seasons to prevent artificial variance inflation."""
        cut = self.min_sets if min_sets is None else min_sets
        return df[df['sets_played'] >= cut].copy()

    @staticmethod
    def _assert_identity_matches_observed(raw_match_df, simulated_season):
        """Require identity blocks to reproduce every observed player metric."""
        observed = calculate_tier1_metrics(
            raw_match_df.copy(deep=True),
            round_digits=None,
        ).sort_values('player_id').reset_index(drop=True)
        simulated = (simulated_season
                     .sort_values('player_id')
                     .reset_index(drop=True))
        pd.testing.assert_frame_equal(
            simulated,
            observed,
            check_exact=True,
            obj='Team-match identity bootstrap output',
        )

    def run_game_level_bootstrap(
        self,
        raw_match_df,
        n_boot=100,
        team_col='team_name',
        match_col='match_id',
        identity=False,
        draw_plan=None,
        season=None,
    ):
        """
        Isolates sampling noise by simulating alternate-reality seasons.
        For each team, resamples its matches with replacement and carries every
        player/set row in a selected team-match together. identity=True selects
        every team-match exactly once, consumes no random draws, and asserts
        exact agreement with the observed player-season metrics.
        """
        if not identity and (
            not isinstance(n_boot, (int, np.integer)) or n_boot < 1
        ):
            raise ValueError("n_boot must be a positive integer")

        if identity:
            print("Bootstrapping: Running team-match identity self-check...")
        else:
            print(f"Bootstrapping: Simulating {n_boot} probabilistic season outcomes...")
        if draw_plan is None:
            if season is None:
                values = (
                    raw_match_df['season'].dropna().unique()
                    if 'season' in raw_match_df else []
                )
                season = int(values[0]) if len(values) == 1 else 0
            draw_plan = build_team_match_draw_plan_for_blocks(
                raw_match_df,
                season=int(season),
                n_boot=n_boot,
                seed=self.seed,
                identity=identity,
                team_column=team_col,
                match_column=match_col,
            )
        expected_replicates = 1 if identity else n_boot
        bootstrap_ids = sorted(draw_plan['bootstrap_id'].unique())
        if len(bootstrap_ids) != expected_replicates:
            raise ValueError('Shared draw plan has the wrong replicate count')

        bootstrapped_seasons = []
        for bootstrap_id in tqdm(bootstrap_ids):
            multiplicities = draw_multiplicities(draw_plan, bootstrap_id)
            resampled_matches = sample_player_set_rows(
                raw_match_df,
                multiplicities,
                team_column=team_col,
            )

            # Aggregate the simulated matches into season totals
            simulated_season = calculate_tier1_metrics(
                resampled_matches,
                round_digits=None,
            )
            if identity:
                self._assert_identity_matches_observed(
                    raw_match_df,
                    simulated_season,
                )
            simulated_season['bootstrap_id'] = bootstrap_id

            bootstrapped_seasons.append(simulated_season)

        return pd.concat(bootstrapped_seasons, ignore_index=True)

    def player_season_bootstrap_variance(self, reps, observed_totals):
        """
        BV[X_spm]: bootstrap variance of every metric for every player-season.
        Observed eligibility is fixed at min_sets before resampling; no second
        sets-played threshold is applied to simulated seasons. Returns BV and a
        long-form finite-replicate coverage table.
        """
        keys = ['season', 'player_id']
        required_reps = keys + self.target_metrics
        missing_reps = [column for column in required_reps if column not in reps.columns]
        if missing_reps:
            raise ValueError(
                f"Bootstrap replicates are missing required columns: {missing_reps}"
            )

        qualifying = self._filter_qualifiers(observed_totals)
        require_unique_rows(
            qualifying,
            keys,
            'Observed qualifying player-season totals',
        )

        qualifying_values = qualifying[keys + self.target_metrics]
        reps_qual = reps.merge(
            qualifying_values[keys],
            on=keys,
            how='inner',
            validate='many_to_one',
        )

        grouped = reps_qual.groupby(keys)[self.target_metrics]
        bv = grouped.var(ddof=1)
        finite_counts = grouped.count().reset_index().melt(
            id_vars=keys,
            value_vars=self.target_metrics,
            var_name='metric',
            value_name='finite_replicates',
        )
        observed_long = qualifying_values.melt(
            id_vars=keys,
            value_vars=self.target_metrics,
            var_name='metric',
            value_name='observed_value',
        )
        coverage = observed_long.merge(
            finite_counts,
            on=keys + ['metric'],
            how='left',
            validate='one_to_one',
        )
        coverage['finite_replicates'] = (
            coverage['finite_replicates'].fillna(0).astype(int)
        )
        coverage['used_for_bv'] = coverage['observed_value'].notna()

        insufficient = coverage[
            coverage['used_for_bv'] & (coverage['finite_replicates'] < 2)
        ]
        if not insufficient.empty:
            examples = insufficient[
                keys + ['metric', 'finite_replicates']
            ].head(10).to_dict('records')
            raise ValueError(
                "At least two finite bootstrap values are required for every "
                f"observed player-season-metric; insufficient examples: {examples}"
            )

        return bv, coverage

    def corrected_stability_ci(self, multi, n_boot=2000):
        """Percentile CI from a full player-cluster bootstrap."""
        cluster_size, cluster_sum, cluster_mean = [], [], []
        cluster_x2_minus_bv, cluster_num = [], []

        for _, g in multi.groupby('player_id'):
            x = g['x'].to_numpy()
            player_bv = g['bv'].to_numpy()
            s_p = len(x)

            # These sufficient statistics retain every season row in the
            # cluster while making 2000 full-estimator recomputations cheap.
            cluster_size.append(s_p)
            cluster_sum.append(x.sum())
            cluster_mean.append(x.mean())
            cluster_x2_minus_bv.append(np.mean(x ** 2 - player_bv))
            cluster_num.append(
                ((x - x.mean()) ** 2).sum() / (s_p - 1) - player_bv.mean()
            )

        cluster_size = np.asarray(cluster_size)
        cluster_sum = np.asarray(cluster_sum)
        cluster_mean = np.asarray(cluster_mean)
        cluster_x2_minus_bv = np.asarray(cluster_x2_minus_bv)
        cluster_num = np.asarray(cluster_num)

        n_players = len(cluster_size)
        sampled_idx = self._ci_rng.integers(
            0, n_players, size=(n_boot, n_players)
        )

        # Duplicate indices are distinct sampled clusters. The grand mean is
        # the total mean over every season row in those sampled clusters.
        bootstrap_grand_mean = (
            cluster_sum[sampled_idx].sum(axis=1)
            / cluster_size[sampled_idx].sum(axis=1)
        )
        bootstrap_num = cluster_num[sampled_idx].mean(axis=1)

        # For each sampled player, mean((x - grand_mean)^2 - BV) equals
        # mean(x^2 - BV) - 2*grand_mean*mean(x) + grand_mean^2.
        bootstrap_den = (
            cluster_x2_minus_bv[sampled_idx]
            - 2 * bootstrap_grand_mean[:, None] * cluster_mean[sampled_idx]
            + bootstrap_grand_mean[:, None] ** 2
        ).mean(axis=1)

        valid = np.isfinite(bootstrap_den) & (bootstrap_den > 0)
        if valid.sum() < 200:
            return np.nan, np.nan

        bootstrap_stability = 1 - bootstrap_num[valid] / bootstrap_den[valid]
        return tuple(np.percentile(bootstrap_stability, [2.5, 97.5]))

    def extract_meta_metrics(self, master_totals, bv):
        """
        Discrimination (paper Eq. 2) and Stability (paper Section 3 estimator).

        Stability follows the published form: BV[X_spm] is subtracted per
        player-season inside the sums, the numerator and denominator run over
        the same multi-season players, and each player is weighted equally.
        Two normalizations of the within-player variance are computed:
          Stability_paper:     1/S_p (paper-exact, ddof=0)
          Stability_corrected: 1/(S_p - 1) (unbiased; removes the -nu/S_p
                               finite-sample bias that matters at S_p = 2-3)
        """
        qual = self._filter_qualifiers(master_totals)
        seasons = sorted(qual['season'].unique())
        results = []

        for metric in self.target_metrics:
            if metric not in qual.columns:
                continue

            bvm = bv[metric].rename('bv').reset_index()
            df = (qual[['season', 'player_id', metric]]
                  .rename(columns={metric: 'x'})
                  .merge(bvm, on=['season', 'player_id'], how='left')
                  .dropna(subset=['x', 'bv']))

            # 1. Discrimination per season: population noise vs single-season
            # spread over the same players (both restricted to rows with x and
            # bv observed, keeping the two populations identical)
            disc = {}
            for s in seasons:
                ds = df[df['season'] == s]
                sv = ds['x'].var(ddof=0)
                disc[s] = 1 - (ds['bv'].mean() / sv) if pd.notna(sv) and sv > 0 else np.nan

            # 2. Stability: common population of players with >= 2 valid seasons
            season_counts = df.groupby('player_id')['x'].transform('size')
            multi = df[season_counts >= 2]

            stab_paper = stab_corrected = stab_lo = stab_hi = np.nan
            if not multi.empty and multi['player_id'].nunique() >= 2:
                grand_mean = multi['x'].mean()
                num_paper, num_corrected, den = [], [], []
                for _, g in multi.groupby('player_id'):
                    s_p = len(g)
                    dev_own = (g['x'] - g['x'].mean()) ** 2
                    num_paper.append((dev_own - g['bv']).mean())
                    num_corrected.append(dev_own.sum() / (s_p - 1) - g['bv'].mean())
                    den.append(((g['x'] - grand_mean) ** 2 - g['bv']).mean())

                den_mean = np.mean(den)
                if den_mean > 0:
                    stab_paper = 1 - np.mean(num_paper) / den_mean
                    stab_corrected = 1 - np.mean(num_corrected) / den_mean

                    # The [0, 1] bound constrains the estimand, not a finite-
                    # sample estimate: for a near-zero-drift metric an unbiased
                    # estimate fluctuates around 1. Resample players as clusters,
                    # carry every selected player's seasons together, and
                    # recompute the full estimator for every draw.
                    stab_lo, stab_hi = self.corrected_stability_ci(multi, n_boot=2000)

            results.append({
                'Metric': metric,
                **{
                    f'Discrimination_{season}': disc.get(season, np.nan)
                    for season in seasons
                },
                'Discrimination': disc.get(seasons[-1], np.nan),
                'Discrimination_avg': np.nanmean(list(disc.values())),
                'Stability_paper': stab_paper,
                'Stability_corrected': stab_corrected,
                'Stab_CI_low': stab_lo,
                'Stab_CI_high': stab_hi,
                'n_multi_season': multi['player_id'].nunique(),
            })

        return pd.DataFrame(results)

    def league_drift_check(self, master_totals):
        """
        Empirical check on sigma^2_SM (league-average drift), which the paper
        treats as negligible and folds into instability. Reports each metric's
        season-level league means and the variance of those means as a share
        of the average within-season (between-player) variance.
        """
        qual = self._filter_qualifiers(master_totals)
        seasons = sorted(qual['season'].unique())
        rows = []
        for metric in self.target_metrics:
            season_means = qual.groupby('season')[metric].mean()
            within_var = qual.groupby('season')[metric].var(ddof=1).mean()
            drift_var = season_means.var(ddof=1)
            rows.append({
                'Metric': metric,
                **{f'mean_{s}': season_means.get(s, np.nan) for s in seasons},
                'drift_share_of_SV': drift_var / within_var if pd.notna(within_var) and within_var > 0 else np.nan,
            })
        return pd.DataFrame(rows)

    def plot_reliability_scatter(self, results_df):
        """Generates the Metric Reliabilities scatter plot for publication."""
        # Drop any metrics that failed to calculate due to missing data/zero variance
        valid_results = results_df.dropna(subset=['Discrimination', 'Stability'])

        plt.figure(figsize=(12, 9))

        x = valid_results['Discrimination']
        y = valid_results['Stability']
        labels = valid_results['Metric']

        plt.scatter(x, y, alpha=0.0)

        # Plot labels dynamically
        for i, label in enumerate(labels):
            plt.text(x.iloc[i], y.iloc[i], label, fontsize=10,
                     ha='center', va='center', weight='bold')

        plt.title('Metric Reliabilities', fontsize=16, pad=15)
        plt.xlabel('Discrimination', fontsize=14)
        plt.ylabel('Stability', fontsize=14)

        # Never clip points silently: extend the frame to the data and mark the
        # theoretical bound when an estimate exceeds it by sampling error
        plt.xlim(max(0.0, x.min() - 0.1), min(1.0, x.max() + 0.1))
        plt.ylim(max(0.0, y.min() - 0.1), max(1.0, y.max() + 0.05))
        if y.max() > 1.0:
            plt.axhline(1.0, color='gray', linewidth=0.8, linestyle=':')

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('Figure_1_Reliability.png', dpi=300)
        print("\nSaved analytical plot: 'Figure_1_Reliability.png'")

# ==========================================
# EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    print("Initializing Meta-Reliability Engine...\n")

    SEASONS = SUPPORTED_MLV_SEASONS
    N_BOOT = 100

    metrics_to_test = list(CONVENTIONAL_METRICS)

    engine = MetaReliabilityEngine(
        target_metrics=metrics_to_test,
        min_sets=20,
        seed=42,
    )

    # 1. Load match-level event data and aggregate observed season totals
    raw, schedules, season_totals = {}, {}, {}
    for season in SEASONS:
        canonical = build_mlv_canonical_season(season)
        raw[season] = canonical.player_sets
        schedules[season] = canonical.schedule
        totals = calculate_tier1_metrics(raw[season], round_digits=None)
        totals = apply_reliability_attempt_eligibility(
            totals,
            raw[season],
        )
        totals['season'] = season
        season_totals[season] = totals

    master_totals = pd.concat(season_totals.values(), ignore_index=True)
    require_unique_rows(
        master_totals,
        ['season', 'player_id'],
        'Observed player-season totals',
    )

    # 2. Bootstrap every season so each player-season gets its own BV[X_spm]
    all_reps = []
    for season in SEASONS:
        # This consumes no random draws and must reproduce observed totals
        # exactly before the stochastic team-match bootstrap is allowed to run.
        identity_plan = build_team_match_draw_plan(
            schedules[season], n_boot=1, seed=42, identity=True
        )
        stochastic_plan = build_team_match_draw_plan(
            schedules[season], n_boot=N_BOOT, seed=42
        )
        engine.run_game_level_bootstrap(
            raw[season], identity=True, draw_plan=identity_plan
        )
        reps = engine.run_game_level_bootstrap(
            raw[season], n_boot=N_BOOT, draw_plan=stochastic_plan
        )
        reps['season'] = season
        all_reps.append(reps)
    all_reps = pd.concat(all_reps, ignore_index=True)
    require_unique_rows(
        all_reps,
        ['season', 'player_id', 'bootstrap_id'],
        'Bootstrap player-season replicates',
    )

    bv, bootstrap_coverage = engine.player_season_bootstrap_variance(
        all_reps,
        master_totals,
    )
    coverage_summary = (bootstrap_coverage[bootstrap_coverage['used_for_bv']]
                        .groupby('metric')['finite_replicates']
                        .agg(['min', 'median', 'max'])
                        .reindex(metrics_to_test))
    print("\n--- FINITE BOOTSTRAP VALUES USED FOR BV ---")
    print(coverage_summary.to_string())

    # 3. Extract Meta-Metrics (paper-exact and bias-corrected stability)
    print("\nExtracting Variance Components...")
    reliability_df = engine.extract_meta_metrics(master_totals, bv)

    print(f"\n--- STABILITY & DISCRIMINATION RESULTS (Discrimination: {SEASONS[-1]} snapshot) ---")
    print(reliability_df.round(3).to_string(index=False))

    # 4. League-average drift check (sigma^2_SM, treated as negligible by the paper)
    print("\n--- LEAGUE-AVERAGE DRIFT CHECK (variance of season means / within-season variance) ---")
    drift_df = engine.league_drift_check(master_totals)
    print(drift_df.round(3).to_string(index=False))

    # Machine-readable diagnostics are generated artifacts (gitignored), not
    # frozen source data. They make every reported value and coverage assertion
    # independently auditable without parsing console output.
    output_dir = Path('generated/reliability')
    output_dir.mkdir(parents=True, exist_ok=True)
    reliability_df.to_csv(output_dir / 'reliability_results.csv', index=False)
    drift_df.to_csv(output_dir / 'league_drift.csv', index=False)
    bootstrap_coverage.to_csv(
        output_dir / 'bootstrap_coverage.csv', index=False
    )
    bv.reset_index().to_csv(
        output_dir / 'bootstrap_variance.csv', index=False
    )
    master_totals.to_csv(
        output_dir / 'observed_player_seasons.csv', index=False
    )

    # 5. Output Publication Visuals using the bias-corrected stability, which
    # keeps E[numerator] = drift >= 0 at S_p = 2-3 where the paper-exact
    # 1/S_p normalization is biased low
    figure_df = reliability_df.rename(columns={'Stability_corrected': 'Stability'})
    engine.plot_reliability_scatter(figure_df)
