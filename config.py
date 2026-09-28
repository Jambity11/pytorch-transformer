def get_config():
    return {
        "dataset_file": None,
        "max_samples": None,
        "epochs": 20,
        "batch_size": 16,
        "seq_len": 128,
        "d_model": 256,
        "layers": 4,
        "heads": 8,
        "d_ff": 1024,
        "lr": 1e-4,
        "seed": 42,
        "eval_samples": 100,
    }
