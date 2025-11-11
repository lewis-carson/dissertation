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
from dataclasses import dataclass


# HalfKP feature dimensions (matching nnue-pytorch reference implementation)
NUM_SQUARES = 64
NUM_PIECE_TYPES = 10  # 5 piece types (P, N, B, R, Q) * 2 colours, kings excluded
NUM_PLANES = NUM_SQUARES * NUM_PIECE_TYPES + 1  # extra plane for bias bucket
HALFKP_FEATURES = NUM_PLANES * NUM_SQUARES  # 64 * (64 * 10 + 1) = 41_024


def orient(is_white_pov: bool, square: int) -> int:
    """Mirror board for black perspective to share weights."""
    return (63 * (not is_white_pov)) ^ square


def halfkp_index(is_white_pov: bool, king_bucket: int, square: int, piece: chess.Piece) -> int:
    """Compute HalfKP feature index, aligned with nnue-pytorch layout."""
    piece_index = (piece.piece_type - 1) * 2 + int(piece.color != is_white_pov)
    return 1 + orient(is_white_pov, square) + piece_index * NUM_SQUARES + king_bucket * NUM_PLANES


@dataclass
class LossParams:
    in_offset: float = 270.0
    out_offset: float = 270.0
    in_scaling: float = 340.0
    out_scaling: float = 380.0
    start_lambda: float = 1.0
    end_lambda: float = 1.0
    pow_exp: float = 2.5
    qp_asymmetry: float = 0.0


def get_halfkp_features(board: chess.Board, perspective: chess.Color) -> list[int]:
    """Return active HalfKP feature indices for the given perspective."""
    indices: list[int] = []

    king_square = board.king(perspective)
    if king_square is None:
        return indices

    king_bucket = orient(perspective == chess.WHITE, king_square)

    for square, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        idx = halfkp_index(perspective == chess.WHITE, king_bucket, square, piece)
        indices.append(idx)

    return indices


def process_position(row):
    """Process a single position row and return HalfKP feature data."""
    try:
        board = chess.Board(row["fen"])

        raw_score = float(row["score"])
        score_cp = raw_score if board.turn == chess.WHITE else -raw_score

        result_raw = int(row.get("result", 0))
        outcome = 1.0 if result_raw > 0 else 0.0 if result_raw < 0 else 0.5

        ply = int(row.get("ply", 0))

        white_features = get_halfkp_features(board, chess.WHITE)
        black_features = get_halfkp_features(board, chess.BLACK)

        if not white_features:
            white_features = [0]
        if not black_features:
            black_features = [0]

        return {
            "white_features": white_features,
            "black_features": black_features,
            "score": score_cp,
            "outcome": outcome,
            "ply": ply,
        }
    except Exception:
        return None


class StreamingChessDataset(IterableDataset):
    """Streaming dataset that converts binpack files on the fly."""

    def __init__(self, binpack_files, converter_path, shuffle_files=True, random_skip=1):
        super().__init__()
        self.shuffle_files = shuffle_files
        self.converter_path = Path(converter_path)
        self.binpack_files = [Path(path) for path in binpack_files]
        self.random_skip = max(1, int(random_skip))

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
                    if self.random_skip > 1 and np.random.randint(self.random_skip) != 0:
                        continue

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

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            fen, score_str, ply_str, result_str = parts[:4]

            yield {
                "fen": fen,
                "score": score_str,
                "ply": ply_str,
                "result": result_str,
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
    """Collate sparse HalfKP features into embedding-bag friendly format."""
    batch_size = len(batch)

    white_indices: list[int] = []
    black_indices: list[int] = []
    white_offsets: list[int] = []
    black_offsets: list[int] = []
    scores: list[float] = []
    outcomes: list[float] = []

    for item in batch:
        white_offsets.append(len(white_indices))
        black_offsets.append(len(black_indices))

        white_indices.extend(item["white_features"] or [0])
        black_indices.extend(item["black_features"] or [0])

        scores.append(item["score"])
        outcomes.append(item["outcome"])

    if not white_indices:
        white_indices = [0]
    if not black_indices:
        black_indices = [0]

    white_indices_tensor = torch.tensor(white_indices, dtype=torch.long)
    black_indices_tensor = torch.tensor(black_indices, dtype=torch.long)
    white_offsets_tensor = torch.tensor(white_offsets, dtype=torch.long)
    black_offsets_tensor = torch.tensor(black_offsets, dtype=torch.long)
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    outcome_tensor = torch.tensor(outcomes, dtype=torch.float32)
    return (
        white_indices_tensor,
        white_offsets_tensor,
        black_indices_tensor,
        black_offsets_tensor,
        score_tensor,
        outcome_tensor,
    )


class HalfKPNetwork(nn.Module):
    """HalfKP neural network using embedding bags to mirror nnue-pytorch flow."""

    def __init__(self, hidden1=256, hidden2=32, hidden3=32):
        super().__init__()

        num_embeddings = HALFKP_FEATURES + 1  # extra row for padding index 0

        self.ft_white = nn.EmbeddingBag(num_embeddings, hidden1, mode="sum")
        self.ft_black = nn.EmbeddingBag(num_embeddings, hidden1, mode="sum")

        self.fc1 = nn.Linear(hidden1 * 2, hidden2)
        self.fc2 = nn.Linear(hidden2, hidden3)
        self.fc3 = nn.Linear(hidden3, 1)

        self.activation = nn.ReLU()

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.EmbeddingBag):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                with torch.no_grad():
                    module.weight[0].zero_()
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, white_indices, white_offsets, black_indices, black_offsets):
        white_transformed = self.activation(
            self.ft_white(white_indices, white_offsets)
        )
        black_transformed = self.activation(
            self.ft_black(black_indices, black_offsets)
        )

        x = torch.cat([white_transformed, black_transformed], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

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


def compute_loss(predictions, scores, outcomes, loss_params: LossParams, epoch_progress: float):
    """Recreate nnue-pytorch blended loss between search scores and game outcomes."""
    p = loss_params

    scorenet = predictions

    q = (scorenet - p.in_offset) / p.in_scaling
    qm = (-scorenet - p.in_offset) / p.in_scaling
    qf = 0.5 * (1.0 + torch.sigmoid(q) - torch.sigmoid(qm))

    s = (scores - p.out_offset) / p.out_scaling
    sm = (-scores - p.out_offset) / p.out_scaling
    pf = 0.5 * (1.0 + torch.sigmoid(s) - torch.sigmoid(sm))

    actual_lambda = p.start_lambda + (p.end_lambda - p.start_lambda) * epoch_progress
    pt = pf * actual_lambda + outcomes * (1.0 - actual_lambda)

    loss = torch.pow(torch.abs(pt - qf), p.pow_exp)

    if p.qp_asymmetry != 0.0:
        loss = loss * ((qf > pt).float() * p.qp_asymmetry + 1.0)

    return loss.mean()


def train_epoch(
    model,
    dataloader,
    optimizer,
    loss_params: LossParams,
    device,
    epoch_idx: int,
    total_epochs: int,
    is_streaming: bool = False,
):
    """Train for one epoch using nnue-pytorch style blended loss."""

    model.train()
    total_loss = 0.0
    num_batches = 0
    epoch_start_time = time.time()
    batch_times = []

    epoch_progress = epoch_idx / max(1, total_epochs - 1)

    for batch_idx, batch in enumerate(dataloader):
        (
            white_indices,
            white_offsets,
            black_indices,
            black_offsets,
            scores,
            outcomes,
        ) = batch

        batch_start_time = time.time()

        white_indices = white_indices.to(device)
        white_offsets = white_offsets.to(device)
        black_indices = black_indices.to(device)
        black_offsets = black_offsets.to(device)
        scores = scores.to(device)
        outcomes = outcomes.to(device)

        optimizer.zero_grad()
        predictions = model(white_indices, white_offsets, black_indices, black_offsets)

        loss = compute_loss(predictions, scores, outcomes, loss_params, epoch_progress)

        if not torch.isfinite(loss):
            print(f"  Warning: Non-finite loss at batch {batch_idx}, skipping...")
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        batch_size = outcomes.size(0)
        samples_per_sec = batch_size / batch_time if batch_time > 0 else 0

        wandb.log(
            {
                "batch_loss": loss.item(),
                "batch": batch_idx,
                "batch_time_sec": batch_time,
                "samples_per_sec": samples_per_sec,
            }
        )

        if batch_idx % 500 == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            avg_batch_time = np.mean(batch_times[-100:]) if batch_times else 0.0
            if is_streaming:
                print(
                    "  Batch {:d}, Avg Loss: {:.6f}, Current Loss: {:.6f}, Batch Time: {:.3f}s, Samples/sec: {:.0f}".format(
                        batch_idx,
                        avg_loss,
                        loss.item(),
                        batch_time,
                        samples_per_sec,
                    )
                )
            else:
                print(
                    "  Batch {}/{} Loss: {:.6f}, Batch Time: {:.3f}s, Samples/sec: {:.0f}".format(
                        batch_idx,
                        len(dataloader),
                        loss.item(),
                        batch_time,
                        samples_per_sec,
                    )
                )

    epoch_duration = time.time() - epoch_start_time
    wandb.log({"epoch_time_sec": epoch_duration})

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, loss_params: LossParams, device, epoch_idx: int, total_epochs: int):
    """Validate the model using the same blended loss as training."""

    model.eval()
    total_loss = 0.0
    num_batches = 0
    epoch_progress = epoch_idx / max(1, total_epochs - 1)

    with torch.no_grad():
        for batch in dataloader:
            (
                white_indices,
                white_offsets,
                black_indices,
                black_offsets,
                scores,
                outcomes,
            ) = batch

            white_indices = white_indices.to(device)
            white_offsets = white_offsets.to(device)
            black_indices = black_indices.to(device)
            black_offsets = black_offsets.to(device)
            scores = scores.to(device)
            outcomes = outcomes.to(device)

            predictions = model(
                white_indices,
                white_offsets,
                black_indices,
                black_offsets,
            )
            loss = compute_loss(predictions, scores, outcomes, loss_params, epoch_progress)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train HalfKP chess neural network')
    parser.add_argument('data_dir', nargs='?', help='Directory containing binpack training data')
    parser.add_argument('--binpack-converter', default=None,
                        help='Path to an existing binpack-reader binary to use for streaming conversion.')
    args = parser.parse_args()
    
    # Hyperparameters
    BATCH_SIZE = 8192
    LEARNING_RATE = 8.75e-4
    NUM_EPOCHS = 600
    GAMMA = 0.992
    NUM_WORKERS = 0
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

    if THREADS > 0:
        torch.set_num_threads(THREADS)
    
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
    val_files = binpack_paths_shuffled[:1]
    print(f"Using {len(val_files)} randomly selected file(s) for validation set")
    
    val_dataset = ChessDataset(val_files, converter_path=converter_path, max_samples=VAL_SAMPLES)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() or torch.backends.mps.is_available(),
    )
    
    # Create streaming training dataset (all files)
    print("\nCreating streaming training dataset...")
    train_dataset = StreamingChessDataset(
        binpack_paths,
        converter_path=converter_path,
        shuffle_files=True,
        random_skip=RANDOM_FEN_SKIPPING,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available() or torch.backends.mps.is_available(),
    )
    
    print(f"Validation set size: {len(val_dataset)}")
    
    # Initialize model
    model = HalfKPNetwork().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_params = LossParams(start_lambda=START_LAMBDA, end_lambda=END_LAMBDA)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=GAMMA)
    
    print("[CHECKPOINT] Model and optimizer initialized")
    sys.stdout.flush()
    
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
        sys.stdout.flush()
        
        print(f"[CHECKPOINT] About to call train_epoch")
        sys.stdout.flush()
        
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_params,
            device,
            epoch,
            NUM_EPOCHS,
            is_streaming=True,
        )
        print(f"[CHECKPOINT] train_epoch returned, train_loss={train_loss:.6f}")
        sys.stdout.flush()
        
        val_loss = validate(
            model,
            val_loader,
            loss_params,
            device,
            epoch,
            NUM_EPOCHS,
        )
        print(f"[CHECKPOINT] validate returned, val_loss={val_loss:.6f}")
        sys.stdout.flush()
        
        print(f"\nEpoch {epoch+1} Summary - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        sys.stdout.flush()

        
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_progress = epoch / max(1, NUM_EPOCHS - 1)
        current_lambda = loss_params.start_lambda + (loss_params.end_lambda - loss_params.start_lambda) * epoch_progress

        wandb.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": current_lr,
                "lambda": current_lambda,
            }
        )

        scheduler.step()
        
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
