import argparse
import json
from plotter import plot_metric_per_time_or_iter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    args = parser.parse_args()

    with open(args.result_file, 'r') as f:
        synthetic_expr_results = json.load(f)

    # STK vs Iteration
    plot_metric_per_time_or_iter(
        result_stats=synthetic_expr_results,
        order=['ScanBestOrder', 'ScanWorstOrder', 'EpsGreedy', 'UCB', 'UniformExploration', 'UniformSample'],
        metric='STK',
        x_axis='iteration',
        ylabel='Sum of Top-k (STK)',
        xlabel='Iteration',
        filename=args.output_dir + 'synthetic_stk_vs_iter.pdf',
        linewidth=2
    )

    # Precision@K vs Iteration
    plot_metric_per_time_or_iter(
        result_stats=synthetic_expr_results,
        order=['ScanBestOrder', 'ScanWorstOrder', 'EpsGreedy', 'UCB', 'UniformExploration', 'UniformSample'],
        metric='Precision@K',
        x_axis='iteration',
        ylabel='Precision@K',
        xlabel='Iteration',
        filename=args.output_dir + 'synthetic_precision_vs_iter.pdf',
        linewidth=2
    )

    # Ablation study (Precision@K vs Iteration)
    plot_metric_per_time_or_iter(
        result_stats=synthetic_expr_results,
        order=['EpsGreedy', 'EpsGreedy (No Rebinning)', 'EpsGreedy (No Subtraction)', 'EpsGreedy (No Fallback)'],
        metric='Precision@K',
        x_axis='iteration',
        ylabel='Precision@K',
        xlabel='Iteration',
        yrange=[0.75, 1.05],
        filename=args.output_dir + 'synthetic_ablation_study.pdf',
        linewidth=2
    )


if __name__ == '__main__':
    main()