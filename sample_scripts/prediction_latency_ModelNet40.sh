python prediction_latency_ModelNet40.py \
-data_dir data/modelnet_parsed/ \
-latency_results_fp data/results/prediction_latency_ModelNet40.pickle \
-n_latency_runs 1 \
-n_features 100 \
-n_threads 4 \
-max_L 5 \
-n_samples 10 \
-n_deltas 1024 \
-n_radial_params 3