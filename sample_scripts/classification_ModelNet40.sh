python classification_ModelNet40.py \
    -data_dir data/modelnet_parsed/ \
    -n_features 100 \
    -n_cores 4 \
    -chunksize 200 \
    -weight_variance 2.0 \
    -max_L 2 \
    -n_train 100 \
    -n_test 100 \
    -bump_width 0.75 \
    -l2_reg 0.01 0.1 1.0 10. \
    -n_radial_params 3 \
    -max_n_deltas 100 \
    -max_radial_param 1. \
    -standardize_matrix_cols \
    -results_fp data/results/classification_ModelNet40.txt \
    -serialize_dir data/results/classification_ModelNet40_scratch/ \
    -train_class z \
    -test_class z