# Audit findings:
# - Enhanced runs append rows to results/enhanced_results.csv with columns:
#   dataset,pred_len,config,MSE,MAE.
# - Dataset names and horizons mirror the run scripts in this repository.

import pandas as pd


def main():
    df = pd.read_csv('results/enhanced_results.csv')

    print("\n" + "=" * 80)
    print("PiXTime Enhancement Results - MSE / MAE Comparison")
    print("=" * 80)

    configs = [
        'baseline',
        'contextual_ve',
        'var_relation',
        'adaptive_patch',
        'multiscale_patch',
        'all_improvements',
    ]

    for dataset in df['dataset'].unique():
        print(f"\n### Dataset: {dataset}")
        dset = df[df['dataset'] == dataset]

        for pred_len in sorted(dset['pred_len'].unique()):
            row = dset[dset['pred_len'] == pred_len]
            baseline_mse = row[row['config'] == 'baseline']['MSE'].values
            if len(baseline_mse) == 0:
                continue
            baseline_mse = baseline_mse[0]

            print(f"\n  Horizon T={pred_len}")
            print(f"  {'Config':<22} {'MSE':>8} {'MAE':>8} {'vs baseline':>12}")
            print(f"  {'-' * 54}")

            for cfg in configs:
                r = row[row['config'] == cfg]
                if len(r) == 0:
                    continue
                mse = r['MSE'].values[0]
                mae = r['MAE'].values[0]
                delta = (mse - baseline_mse) / baseline_mse * 100
                marker = ' *' if delta < 0 else '  '
                print(f"  {cfg:<22} {mse:>8.4f} {mae:>8.4f} {delta:>+10.2f}%{marker}")

    print("\n" + "=" * 80)
    print("Average MSE Improvement over Baseline (%)")
    print("=" * 80)

    for cfg in configs[1:]:
        merged = df.merge(
            df[df['config'] == 'baseline'][['dataset', 'pred_len', 'MSE']],
            on=['dataset', 'pred_len'],
            suffixes=('', '_base'),
        )
        merged = merged[merged['config'] == cfg]
        if len(merged) == 0:
            continue
        avg_imp = ((merged['MSE_base'] - merged['MSE']) / merged['MSE_base'] * 100).mean()
        print(f"  {cfg:<25}: {avg_imp:>+.2f}%")


if __name__ == '__main__':
    main()
