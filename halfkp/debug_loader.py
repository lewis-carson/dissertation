
import torch
import binpack_loader
import sys
import os

# Mock config
class SkipConfig:
    def __init__(self):
        self.filtered = False
        self.random_fen_skipping = 0
        self.wld_filtered = False
        self.early_fen_skipping = -1
        self.simple_eval_skipping = -1
        self.param_index = 0
    
    def to_dict(self):
        return {
            "filtered": self.filtered,
            "random_fen_skipping": self.random_fen_skipping,
            "wld_filtered": self.wld_filtered,
            "early_fen_skipping": self.early_fen_skipping,
            "simple_eval_skipping": self.simple_eval_skipping,
            "param_index": self.param_index,
        }

def test_loader():
    # Find a binpack file
    data_dir = "binpack-rust-main/test"
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".binpack")]
    if not files:
        print("No binpack files found in ../test")
        return

    print(f"Testing with file: {files[0]}")
    
    batch_size = 4
    # Use SparseBatchStream directly as in train.py
    stream = binpack_loader.SparseBatchStream(
        feature_set="HalfKP",
        files=files,
        batch_size=batch_size,
        skip_config=SkipConfig().to_dict(),
        cyclic=False,
        num_workers=1
    )
    
    # Manually iterate
    for i, batch in enumerate(stream):
        if i > 0: break
        
        # Convert to tensors manually (simplified from train.py)
        # batch is a tuple of numpy arrays
        (
            us,
            them,
            white_indices,
            white_values,
            black_indices,
            black_values,
            outcome,
            score,
            psqt_indices,
            layer_stack_indices,
        ) = batch
        
        print("--- Batch 0 ---")
        print(f"White Indices Shape: {white_indices.shape}")
        print(f"White Indices (first row): {white_indices[0]}")
        print(f"White Values (first row): {white_values[0]}")
        print(f"Score: {score}")
        print(f"Outcome: {outcome}")
        
        # Check for active features
        active_white = white_indices[0][white_indices[0] != -1]
        print(f"Active White Indices: {active_white}")
        print(f"Max Index: {white_indices.max()}")
        print(f"Min Index: {white_indices.min()}")

if __name__ == "__main__":
    test_loader()
