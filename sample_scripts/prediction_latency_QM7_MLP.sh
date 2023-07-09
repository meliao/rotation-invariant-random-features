python prediction_latency_QM7_MLP.py \
        -data_fp data/qm7/qm7.mat \
        -n_features 100 \
        -n_samples 100 \
        -n_threads 4 \
        -width 1_000 \
        -depth 2 \
        -max_L 5 \
        -n_latency_runs 1 \
        -latency_results_fp data/results/prediction_latency_QM7_MLP.pickle