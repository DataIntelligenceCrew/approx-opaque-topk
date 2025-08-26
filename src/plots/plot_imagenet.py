import argparse
from plotter import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt-file', type=str, required=True)
    parser.add_argument('--result-file', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--index-time-file', type=str, required=True)
    parser.add_argument('--imagenet-class-id', type=int, required=True)
    args = parser.parse_args()

    with open(args.result_file, 'r') as f:
        expr_results = json.load(f)

    # STK vs Time
    plot_metric_per_time_or_iter(
        result_stats=expr_results,
        order=['ScanBestOrder', 'ScanWorstOrder', 'EpsGreedy', 'UCB', 'UniformExploration', 'UniformSample'],
        metric='STK',
        x_axis='sec',
        ylabel='Sum of Top-k (STK)',
        xlabel='Time (s)',
        filename=args.output_dir + 'imagenet_' + str(args.imagenet_class_id) + '_stk_vs_time.pdf',
        linewidth=2
    )

    # Precision vs Time
    plot_metric_per_time_or_iter(
        result_stats=expr_results,
        order=['ScanBestOrder', 'ScanWorstOrder', 'EpsGreedy', 'UCB', 'UniformExploration', 'UniformSample'],
        metric='Precision@K',
        x_axis='sec',
        ylabel='Precision@K',
        xlabel='Time (s)',
        filename=args.output_dir + 'imagenet_' + str(args.imagenet_class_id) + '_precision_vs_time.pdf',
        linewidth=2
    )

    # End-to-end latency
    plot_total_latency_stacked(
        result_stats=expr_results,
        order=['EpsGreedy', 'UCB', 'UniformExploration', 'UniformSample'],
        ylabel='End-to-End Latency (s)',
        filename=args.output_dir + 'imagenet_' +  str(args.imagenet_class_id) + '_latency_total.pdf',
        time_unit='s',
        include_index_build_time=True,
        xtick_rotation=20,
        gt_json_path=args.gt_file,
        dendrogram_index_time_txt_path=args.index_time_file
    )

    # Overhead per iteration
    plot_iter_latency_stacked(
        result_stats=expr_results,
        order=['EpsGreedy', 'UCB', 'UniformExploration', 'UniformSample', 'EpsGreedy (No Rebinning)', 'EpsGreedy (No Subtraction)', 'EpsGreedy (No Fallback)'],
        ylabel='Precision@K',
        xlabel='Time (s)',
        filename=args.output_dir + 'imagenet_' + str(args.imagenet_class_id) + '_latency_iter.pdf',
        time_unit='s',
        include_scoring_fn=False
    )

    # Parameter study
    plot_metric_per_time_or_iter(
        result_stats=expr_results,
            order=['EpsGreedy', 'EpsGreedy (F=5%)', 'EpsGreedy (F=10%)', 'EpsGreedy (25%)', 'EpsGreedy (Batch=100)', 'EpsGreedy (Batch=200)', 'EpsGreedy (Batch=800)'],
        metric='Precision@K',
        x_axis='sec',
        ylabel='Precision@K',
        xlabel='Time (s)',
        filename=args.output_dir + 'imagenet_' + str(args.imagenet_class_id) + '_parameter_study.pdf',
        linewidth=2
    )



if __name__ == '__main__':
    main()