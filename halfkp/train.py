import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import chess
import numpy as np
from pathlib import Path
import shutil
import subprocess
import wandb
import sys
import argparse
import time


# HalfKP feature dimensions
# 64 king squares * 10 piece types * 64 squares = 40960 features per side
# We'll use a simpler encoding: king square (64) * piece-square (64 * 12 pieces)
PIECE_TO_INDEX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}

# HalfKP: For each king position, encode all other pieces
# Feature index = king_square * (64 * 10 piece types) + piece_square * 10 + piece_type
# White pieces: 0-4, Black pieces: 5-9
NUM_PIECE_TYPES = 10  # 5 piece types (P,N,B,R,Q) * 2 colors (excluding kings)
NUM_SQUARES = 64
HALFKP_FEATURES = NUM_SQUARES * NUM_SQUARES * NUM_PIECE_TYPES  # 40960


def get_halfkp_features(board, perspective):
    """
    Extract HalfKP features for a given position from a perspective.
    Returns a list of active feature indices.
    """
    features = []
    
    # Find the king square from perspective
    king_square = board.king(perspective)
    if king_square is None:
        return features
    
    # Encode all pieces except the perspective's king
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        # Skip the perspective's king
        if piece.piece_type == chess.KING and piece.color == perspective:
            continue
        
        # Calculate piece type index (0-9)
        # White pieces: 0-4, Black pieces: 5-9
        # NOTE: KING pieces are not included in HalfKP encoding (only the perspective king is used for bucketing)
        if piece.piece_type == chess.KING:
            # Skip enemy king - HalfKP only uses the perspective king for feature bucketing
            continue
        
        base_idx = PIECE_TO_INDEX[piece.piece_type]
        piece_idx = base_idx if piece.color == chess.WHITE else (base_idx + 5)
        
        # Calculate feature index
        # feature = king_square * (64 * 10) + square * 10 + piece_idx
        feature_idx = king_square * (NUM_SQUARES * NUM_PIECE_TYPES) + square * NUM_PIECE_TYPES + piece_idx
        
        # Sanity check
        if feature_idx >= HALFKP_FEATURES:
            continue
            
        features.append(feature_idx)
    
    return features


def process_position(row):
    """Process a single position row and return features."""
    try:
        board = chess.Board(row['fen'])
        # Use score column - normalize it
        raw_score = float(row['score'])
        
        # Handle edge cases and normalize score to [-1, 1] range
        # Scores can range widely, so we use tanh to squash them
        # Divide by a scaling factor to make the range reasonable
        eval_score = np.tanh(raw_score / 1000.0)  # Scale by 1000
        
        # Clip to valid range
        eval_score = np.clip(eval_score, -1.0, 1.0)
        
        # Get features for both sides
        white_features = get_halfkp_features(board, chess.WHITE)
        black_features = get_halfkp_features(board, chess.BLACK)
        
        return {
            'white_features': white_features,
            'black_features': black_features,
            'eval': eval_score,
            'stm': board.turn  # side to move
        }
    except Exception as e:
        # Skip invalid positions
        return None


class StreamingChessDataset(IterableDataset):
    """Streaming dataset that converts binpack files on the fly."""

    def __init__(self, binpack_files, converter_path, shuffle_files=True):
        super().__init__()
        self.shuffle_files = shuffle_files
        self.converter_path = Path(converter_path)
        self.binpack_files = [Path(path) for path in binpack_files]

        if not self.binpack_files:
            raise ValueError("No binpack files provided for streaming dataset")

        print(f"Found {len(self.binpack_files)} binpack file(s) to process")

    def __iter__(self):
        files_to_process = self.binpack_files.copy()
        if self.shuffle_files and len(files_to_process) > 1:
            np.random.shuffle(files_to_process)

        for file_idx, binpack_file in enumerate(files_to_process):
            if file_idx % 10 == 0:
                print(f"Processing file {file_idx + 1}/{len(files_to_process)}: {binpack_file.name}")

            try:
                for row in _stream_binpack_entries(binpack_file, self.converter_path):
                    result = process_position(row)
                    if result is not None:
                        yield result
            except Exception as exc:
                print(f"Error processing {binpack_file}: {exc}")
                continue


class ChessDataset(Dataset):
    """In-memory dataset for validation set (smaller subset)."""

    def __init__(self, binpack_files, converter_path, max_samples=10000):
        self.data = []
        converter_path = Path(converter_path)
        binpack_files = [Path(path) for path in binpack_files]

        print(f"Loading validation data from {len(binpack_files)} file(s)...")

        total_loaded = 0
        for binpack_file in binpack_files:
            if total_loaded >= max_samples:
                break

            try:
                for row in _stream_binpack_entries(binpack_file, converter_path):
                    if total_loaded >= max_samples:
                        break

                    result = process_position(row)
                    if result is not None:
                        self.data.append(result)
                        total_loaded += 1
            except Exception as exc:
                print(f"Error processing {binpack_file}: {exc}")
                continue

        print(f"Successfully loaded {len(self.data)} validation positions")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _gather_paths(base_dir, patterns):
    """Collect files in base_dir matching patterns recursively."""
    collected = []
    for pattern in patterns:
        collected.extend(Path(base_dir).rglob(pattern))

    unique = []
    seen = set()
    for path in collected:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)

    unique.sort()
    return unique


def _locate_binpack_reader(executable_hint=None):
    """Find the compiled binpack-reader binary, building it if needed."""
    if executable_hint:
        candidate = Path(executable_hint)
        if candidate.exists() and candidate.is_file():
            return candidate

    repo_root = Path(__file__).parent.parent
    project_root = repo_root / "binpack-reader"
    candidates = [
        project_root / "target" / "release" / "binpack-reader",
        project_root / "target" / "debug" / "binpack-reader",
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    cargo = shutil.which("cargo")
    if cargo is None:
        return None

    print("binpack-reader binary not found, building with cargo...")
    try:
        subprocess.run(
            [cargo, "build", "--release"],
            cwd=project_root,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="ignore") if exc.stderr else ""
        print("Failed to build binpack-reader with cargo.")
        if stderr:
            print(stderr.strip())
        return None

    built_candidate = project_root / "target" / "release" / "binpack-reader"
    if built_candidate.exists() and built_candidate.is_file():
        return built_candidate

    print("binpack-reader build completed but executable not found.")
    return None


def _stream_binpack_entries(binpack_file, converter_path):
    """Stream positions from a binpack file using the helper binary."""
    binpack_file = Path(binpack_file)
    converter_path = Path(converter_path)

    if not binpack_file.exists():
        raise FileNotFoundError(f"Binpack file not found: {binpack_file}")

    cmd = [
        str(converter_path),
        str(binpack_file.parent),
        str(binpack_file.parent),
        "--stdout",
        "--single-file",
        str(binpack_file),
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    if process.stdout is None:
        process.kill()
        raise RuntimeError("Failed to open pipe to binpack-reader stdout")

    return_code = None
    try:
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue

            try:
                fen, score_str = line.split("\t", 1)
            except ValueError:
                continue

            yield {
                "fen": fen,
                "score": score_str,
            }

        return_code = process.wait()
    finally:
        stdout_stream = process.stdout
        if stdout_stream is not None:
            stdout_stream.close()

        if process.poll() is None:
            process.kill()
            process.wait()
        elif return_code not in (None, 0):
            raise RuntimeError(
                f"binpack-reader exited with status {return_code} while processing {binpack_file}"
            )


def collate_fn(batch):
    """Custom collate function to create dense feature tensors."""
    batch_size = len(batch)
    
    # Create feature lists
    white_features_list = []
    black_features_list = []
    evals = []
    
    for item in batch:
        white_features_list.append(item['white_features'])
        black_features_list.append(item['black_features'])
        evals.append(item['eval'])
    
    # Convert to dense tensors (one-hot encoding)
    white_dense = torch.zeros((batch_size, HALFKP_FEATURES))
    black_dense = torch.zeros((batch_size, HALFKP_FEATURES))
    
    for i, (wf, bf) in enumerate(zip(white_features_list, black_features_list)):
        if wf:
            white_dense[i, wf] = 1.0
        if bf:
            black_dense[i, bf] = 1.0
    
    evals_tensor = torch.tensor(evals, dtype=torch.float32)
    
    return white_dense, black_dense, evals_tensor


class HalfKPNetwork(nn.Module):
    """
    HalfKP neural network architecture similar to NNUE.
    Architecture: 40960x2 -> 256 -> 32 -> 32 -> 1
    """
    
    def __init__(self, input_dim=HALFKP_FEATURES, hidden1=256, hidden2=32, hidden3=32):
        super(HalfKPNetwork, self).__init__()
        
        # Feature transformer (separate for each side)
        self.ft_white = nn.Linear(input_dim, hidden1)
        self.ft_black = nn.Linear(input_dim, hidden1)
        
        # Hidden layers
        self.fc1 = nn.Linear(hidden1 * 2, hidden2)
        self.fc2 = nn.Linear(hidden2, hidden3)
        self.fc3 = nn.Linear(hidden3, 1)
        
        # Use ClippedReLU (ReLU with clipping) as in NNUE
        self.activation = nn.ReLU()
        
        # Initialize weights with small values to prevent explosion
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with small values for numerical stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, white_features, black_features):
        # Feature transformation
        white_transformed = self.activation(self.ft_white(white_features))
        black_transformed = self.activation(self.ft_black(black_features))
        
        # Concatenate both perspectives
        x = torch.cat([white_transformed, black_transformed], dim=1)
        
        # Hidden layers with ReLU activation
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        
        # Output layer with tanh to bound output
        x = torch.tanh(self.fc3(x))
        
        return x.squeeze(-1)


def normalize_eval(eval_score):
    """
    Normalize evaluation scores.
    Input: win probability or raw value
    Output: normalized value suitable for training
    """
    # Assuming eval_score is win probability, convert to a reasonable range
    # Could also use sigmoid-based scaling
    return eval_score


def train_epoch(model, dataloader, optimizer, criterion, device, is_streaming=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    epoch_start_time = time.time()
    batch_times = []
    
    for batch_idx, (white_feat, black_feat, evals) in enumerate(dataloader):
        batch_start_time = time.time()
        
        white_feat = white_feat.to(device)
        black_feat = black_feat.to(device)
        evals = evals.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(white_feat, black_feat)
        
        # Compute loss
        loss = criterion(predictions, evals)
        
        # Check for NaN/inf
        if not torch.isfinite(loss):
            print(f"  Warning: Non-finite loss at batch {batch_idx}, skipping...")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Calculate batch timing metrics
        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        batch_size = white_feat.size(0)
        samples_per_sec = batch_size / batch_time if batch_time > 0 else 0
        
        # Log batch loss and timing to wandb
        wandb.log({
            "batch_loss": loss.item(),
            "batch": batch_idx,
            "batch_time_sec": batch_time,
            "samples_per_sec": samples_per_sec,
        })
        
        if batch_idx % 500 == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            avg_batch_time = np.mean(batch_times[-100:]) if batch_times else 0  # Last 100 batches
            if is_streaming:
                print(f"  Batch {batch_idx}, Avg Loss: {avg_loss:.6f}, Current Loss: {loss.item():.6f}, "
                      f"Batch Time: {batch_time:.3f}s, Samples/sec: {samples_per_sec:.0f}")
            else:
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}, "
                      f"Batch Time: {batch_time:.3f}s, Samples/sec: {samples_per_sec:.0f}")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for white_feat, black_feat, evals in dataloader:
            white_feat = white_feat.to(device)
            black_feat = black_feat.to(device)
            evals = evals.to(device)
            
            predictions = model(white_feat, black_feat)
            loss = criterion(predictions, evals)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train HalfKP chess neural network')
    parser.add_argument('data_dir', nargs='?', help='Directory containing binpack training data')
    parser.add_argument('--binpack-converter', default=None,
                        help='Path to an existing binpack-reader binary to use for streaming conversion.')
    args = parser.parse_args()
    
    # Hyperparameters
    BATCH_SIZE = 16384 // 64
    LEARNING_RATE = 8.75e-4
    NUM_EPOCHS = 600
    GAMMA = 0.992
    NUM_WORKERS = 4
    THREADS = 2
    NETWORK_SAVE_PERIOD = 10
    VALIDATION_SIZE = 1000000
    EPOCH_SIZE = 100000000
    START_LAMBDA = 1.0
    END_LAMBDA = 0.75
    RANDOM_FEN_SKIPPING = 3
    STREAMING = True  # Use streaming dataset for all data
    VAL_SAMPLES = 50000  # Number of positions for validation set
    
    # Device - prioritize Metal (MPS) for Mac, then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")
    print(f"Device: {device}")
    
    # Data directory - use argument or default
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser()
    else:
        data_dir = Path(__file__).parent.parent / "data"

    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")

    print(f"Data directory: {data_dir}")

    binpack_paths = _gather_paths(data_dir, ["*.binpack", "*.no-db.binpack"])
    
    if not binpack_paths:
        raise ValueError(f"No binpack files found in {data_dir}")

    converter_path = _locate_binpack_reader(args.binpack_converter)
    if converter_path is None:
        raise RuntimeError(
            "Could not locate or build binpack-reader binary. Please run "
            "`cargo build --release` in data-hf/binpack-reader or provide an explicit path via --binpack-converter."
        )

    print(f"Using binpack-reader at {converter_path}")
    
    # Create validation set from a random subset of files
    # Randomize file selection to avoid chronological bias
    binpack_paths_shuffled = binpack_paths.copy()
    np.random.shuffle(binpack_paths_shuffled)
    val_files = binpack_paths_shuffled[:max(1, len(binpack_paths) // 20)]  # Use ~5% of files for validation
    print(f"Using {len(val_files)} randomly selected file(s) for validation set")
    
    val_dataset = ChessDataset(val_files, converter_path=converter_path, max_samples=VAL_SAMPLES)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)
    
    # Create streaming training dataset (all files)
    print("\nCreating streaming training dataset...")
    train_dataset = StreamingChessDataset(binpack_paths, converter_path=converter_path, shuffle_files=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             collate_fn=collate_fn, num_workers=0)
    
    print(f"Validation set size: {len(val_dataset)}")
    
    # Initialize model
    model = HalfKPNetwork().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize wandb
    wandb.init(
        project="halfkp-chess",
        config={
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "gamma": GAMMA,
            "num_workers": NUM_WORKERS,
            "threads": THREADS,
            "network_save_period": NETWORK_SAVE_PERIOD,
            "validation_size": VALIDATION_SIZE,
            "epoch_size": EPOCH_SIZE,
            "start_lambda": START_LAMBDA,
            "end_lambda": END_LAMBDA,
            "random_fen_skipping": RANDOM_FEN_SKIPPING,
            "streaming": STREAMING,
            "device": str(device),
            "num_binpack_files": len(binpack_paths),
            "data_dir": str(data_dir),
            "binpack_converter": str(converter_path),
            "binpack_stream_mode": "stdout",
        }
    )
    
    # Training loop
    print("\nStarting training with streaming dataset...")
    print("Note: Training on full dataset across all binpack files")
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, is_streaming=True)
        val_loss = validate(model, val_loader, criterion, device)
        
        print(f"\nEpoch {epoch+1} Summary - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = Path(__file__).parent / f"halfkp_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_path = Path(__file__).parent / "halfkp_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
