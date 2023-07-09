    python prediction_latency_QM7_ridge.py \
        -data_fp data/qm7/qm7.mat \
        -n_features 100 \
        -n_samples 100 \
        -max_L 5 \
        -n_latency_runs 1 \
        -latency_results_fp data/results/prediction_latency_QM7_ridge.pickle