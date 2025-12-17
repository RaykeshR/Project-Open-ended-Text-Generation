CUDA_VISIBLE_DEVICES=0 python compute_coherence.py\
    --opt_model_name facebook/opt-2.7b\
    --test_path $1\
    --result_path $2