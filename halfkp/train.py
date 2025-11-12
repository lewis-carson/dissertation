import argparse
import ctypes
import math
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
import wandb


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


LIB_CANDIDATES = [
    "training_data_loader.dylib",
    "training_data_loader.so",
    "training_data_loader.dll",
]


@dataclass
class DataloaderSkipConfig:
    filtered: bool = False
    random_fen_skipping: int = 0
    wld_filtered: bool = False
    early_fen_skipping: int = -1
    simple_eval_skipping: int = -1
    param_index: int = 0


class CDataloaderSkipConfig(ctypes.Structure):
    _fields_ = [
        ("filtered", ctypes.c_bool),
        ("random_fen_skipping", ctypes.c_int),
        ("wld_filtered", ctypes.c_bool),
        ("early_fen_skipping", ctypes.c_int),
        ("simple_eval_skipping", ctypes.c_int),
        ("param_index", ctypes.c_int),
    ]

    def __init__(self, config: DataloaderSkipConfig):
        super().__init__(
            filtered=config.filtered,
            random_fen_skipping=config.random_fen_skipping,
            wld_filtered=config.wld_filtered,
            early_fen_skipping=config.early_fen_skipping,
            simple_eval_skipping=config.simple_eval_skipping,
            param_index=config.param_index,
        )


def _load_training_library():
    base_dir = Path(__file__).resolve().parent
    for candidate_name in LIB_CANDIDATES:
        candidate_path = base_dir / candidate_name
        if candidate_path.exists():
            return ctypes.cdll.LoadLibrary(str(candidate_path))
    raise FileNotFoundError(
        f"training_data_loader library not found next to {__file__}. Expected one of: {', '.join(LIB_CANDIDATES)}"
    )


def _init_training_library():
    library = _load_training_library()

    library.create_sparse_batch_stream.restype = ctypes.c_void_p
    library.create_sparse_batch_stream.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_int,
        ctypes.c_bool,
        CDataloaderSkipConfig,
    ]

    library.destroy_sparse_batch_stream.argtypes = [ctypes.c_void_p]

    library.fetch_next_sparse_batch.restype = ctypes.c_void_p
    library.fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]

    library.destroy_sparse_batch.argtypes = [ctypes.c_void_p]

    library.get_sparse_batch_from_fens.restype = ctypes.c_void_p
    library.get_sparse_batch_from_fens.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
    ]

    return library


TRAINING_LIB = _init_training_library()


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


class SparseBatch(ctypes.Structure):
    _fields_ = [
        ("num_inputs", ctypes.c_int),
        ("size", ctypes.c_int),
        ("is_white", ctypes.POINTER(ctypes.c_float)),
        ("outcome", ctypes.POINTER(ctypes.c_float)),
        ("score", ctypes.POINTER(ctypes.c_float)),
        ("num_active_white_features", ctypes.c_int),
        ("num_active_black_features", ctypes.c_int),
        ("max_active_features", ctypes.c_int),
        ("white", ctypes.POINTER(ctypes.c_int)),
        ("black", ctypes.POINTER(ctypes.c_int)),
        ("white_values", ctypes.POINTER(ctypes.c_float)),
        ("black_values", ctypes.POINTER(ctypes.c_float)),
        ("psqt_indices", ctypes.POINTER(ctypes.c_int)),
        ("layer_stack_indices", ctypes.POINTER(ctypes.c_int)),
    ]

    def get_tensors(self, device: torch.device):
        def _move(tensor: torch.Tensor) -> torch.Tensor:
            if device.type == "cuda":
                return tensor.pin_memory().to(device=device, non_blocking=True)
            if device.type == "cpu":
                return tensor.clone()
            return tensor.to(device=device)

        white_values = _move(
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.white_values, shape=(self.size, self.max_active_features)
                )
            )
        )
        black_values = _move(
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.black_values, shape=(self.size, self.max_active_features)
                )
            )
        )
        white_indices = _move(
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.white, shape=(self.size, self.max_active_features)
                )
            ).to(dtype=torch.long, copy=True)
        )
        black_indices = _move(
            torch.from_numpy(
                np.ctypeslib.as_array(
                    self.black, shape=(self.size, self.max_active_features)
                )
            ).to(dtype=torch.long, copy=True)
        )
        us = _move(
            torch.from_numpy(
                np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))
            )
        )
        them = 1.0 - us
        outcome = _move(
            torch.from_numpy(
                np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))
            )
        )
        score = _move(
            torch.from_numpy(
                np.ctypeslib.as_array(self.score, shape=(self.size, 1))
            )
        )
        psqt_indices = _move(
            torch.from_numpy(
                np.ctypeslib.as_array(self.psqt_indices, shape=(self.size,))
            ).to(dtype=torch.long, copy=True)
        )
        layer_stack_indices = _move(
            torch.from_numpy(
                np.ctypeslib.as_array(self.layer_stack_indices, shape=(self.size,))
            ).to(dtype=torch.long, copy=True)
        )
        return (
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
        )


def _to_c_str_array(items):
    arr = (ctypes.c_char_p * len(items))()
    arr[:] = [s.encode("utf-8") for s in items]
    return arr


def create_sparse_batch_stream(
    feature_set: str,
    concurrency: int,
    filenames,
    batch_size: int,
    cyclic: bool,
    config: DataloaderSkipConfig,
):
    if len(filenames) == 0:
        raise ValueError("No binpack files provided to sparse loader")
    filenames_bytes = _to_c_str_array(filenames)
    return TRAINING_LIB.create_sparse_batch_stream(
        feature_set.encode("utf-8"),
        max(1, concurrency),
        len(filenames),
        filenames_bytes,
        batch_size,
        bool(cyclic),
        CDataloaderSkipConfig(config),
    )


def fetch_next_sparse_batch(stream_handle):
    raw_ptr = TRAINING_LIB.fetch_next_sparse_batch(stream_handle)
    if not raw_ptr:
        return None
    return ctypes.cast(raw_ptr, ctypes.POINTER(SparseBatch))


def destroy_sparse_batch(batch_handle):
    TRAINING_LIB.destroy_sparse_batch(batch_handle)


def destroy_sparse_batch_stream(stream_handle):
    TRAINING_LIB.destroy_sparse_batch_stream(stream_handle)


class SparseBatchIterableDataset(IterableDataset):
    def __init__(
        self,
        feature_set: str,
        filenames,
        batch_size: int,
        skip_config: DataloaderSkipConfig,
        cyclic: bool,
        num_workers: int,
    ):
        super().__init__()
        self.feature_set = feature_set
        self.filenames = list(filenames)
        self.batch_size = batch_size
        self.skip_config = skip_config
        self.cyclic = cyclic
        self.num_workers = num_workers

    def __iter__(self):
        stream = create_sparse_batch_stream(
            self.feature_set,
            self.num_workers,
            self.filenames,
            self.batch_size,
            self.cyclic,
            self.skip_config,
        )
        device = torch.device("cpu")
        try:
            batch_count = 0
            while True:
                try:
                    batch_ptr = fetch_next_sparse_batch(stream)
                except Exception as e:
                    raise RuntimeError(f"Error fetching batch {batch_count} from C++ dataloader: {e}") from e
                
                if batch_ptr is None:
                    break
                
                try:
                    tensors = batch_ptr.contents.get_tensors(device)
                except Exception as e:
                    raise RuntimeError(f"Error converting batch {batch_count} to tensors: {e}") from e
                finally:
                    destroy_sparse_batch(batch_ptr)
                
                batch_count += 1
                yield tensors
        finally:
            destroy_sparse_batch_stream(stream)


class FixedNumBatchesDataset(Dataset):
    def __init__(self, dataset, num_batches):
        super().__init__()
        self.dataset = dataset
        self.iter = iter(self.dataset)
        self.num_batches = num_batches

        self._prefetch_queue = queue.Queue(maxsize=100)
        self._prefetch_thread = None
        self._stop_prefetching = threading.Event()
        self._prefetch_started = False
        self._lock = threading.Lock()

    def _prefetch_worker(self):
        try:
            while not self._stop_prefetching.is_set():
                try:
                    item = next(self.iter)
                    self._prefetch_queue.put(item)
                except StopIteration:
                    self._prefetch_queue.put(None)
                    break
                except queue.Full:
                    continue
        except Exception as exc:
            self._prefetch_queue.put(exc)

    def _start_prefetching(self):
        with self._lock:
            if not self._prefetch_started:
                self._prefetch_thread = threading.Thread(
                    target=self._prefetch_worker,
                    daemon=True,
                )
                self._prefetch_thread.start()
                self._prefetch_started = True

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        self._start_prefetching()

        try:
            item = self._prefetch_queue.get(timeout=300.0)
            if item is None:
                raise StopIteration("End of dataset reached")
            if isinstance(item, Exception):
                raise item
            return item
        except queue.Empty as exc:
            raise RuntimeError("Prefetch timeout - no data available") from exc

    def __del__(self):
        if hasattr(self, "_stop_prefetching"):
            self._stop_prefetching.set()
        if hasattr(self, "_prefetch_thread") and self._prefetch_thread:
            self._prefetch_thread.join(timeout=1.0)


@dataclass
class PreparedBatch:
    white_indices: torch.Tensor
    white_offsets: torch.Tensor
    white_weights: torch.Tensor
    black_indices: torch.Tensor
    black_offsets: torch.Tensor
    black_weights: torch.Tensor
    scores: torch.Tensor
    outcomes: torch.Tensor
    size: int


def _flatten_sparse(
    indices: torch.Tensor, values: torch.Tensor, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert fixed-width sparse arrays into flattened indices/weights plus offsets."""

    indices = indices.to(dtype=torch.long, copy=True)
    values = values.to(dtype=torch.float32, copy=True)

    mask = values != 0
    lengths = mask.sum(dim=1, dtype=torch.long)
    flat_indices = indices[mask]
    flat_weights = values[mask]

    batch_size = indices.size(0)
    total_active = lengths.sum().item()

    if total_active == 0:
        flat_indices = torch.zeros(1, dtype=torch.long)
        flat_weights = torch.zeros(1, dtype=torch.float32)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        if batch_size > 0:
            lengths[0] = 1

    offsets = torch.zeros(batch_size, dtype=torch.long)
    if batch_size > 1:
        offsets[1:] = torch.cumsum(lengths[:-1], dim=0)

    return (
        flat_indices.to(device=device, non_blocking=True),
        offsets.to(device=device, non_blocking=True),
        flat_weights.to(device=device, non_blocking=True),
    )


def prepare_sparse_batch(batch, device: torch.device) -> PreparedBatch:
    """Transform a SparseBatch tuple into embedding-bag friendly tensors."""

    (
        _us,
        _them,
        white_indices,
        white_values,
        black_indices,
        black_values,
        outcome,
        score,
        _psqt_indices,
        _layer_stack_indices,
    ) = batch

    white_idx, white_off, white_w = _flatten_sparse(white_indices, white_values, device)
    black_idx, black_off, black_w = _flatten_sparse(black_indices, black_values, device)

    scores = score.view(-1).to(device=device, dtype=torch.float32, non_blocking=True)
    outcomes = outcome.view(-1).to(device=device, dtype=torch.float32, non_blocking=True)

    batch_size = white_indices.size(0)

    return PreparedBatch(
        white_indices=white_idx,
        white_offsets=white_off,
        white_weights=white_w,
        black_indices=black_idx,
        black_offsets=black_off,
        black_weights=black_w,
        scores=scores,
        outcomes=outcomes,
        size=batch_size,
    )


def create_sparse_dataloader(
    binpack_files,
    batch_size: int,
    skip_config: DataloaderSkipConfig,
    num_workers: int = 0,
    cyclic: bool = False,
):
    """Build a DataLoader backed by the nnue-pytorch C++ sparse pipeline."""

    files = [str(Path(path)) for path in binpack_files]
    dataset = SparseBatchIterableDataset(
        "HalfKP",
        files,
        batch_size,
        skip_config=skip_config,
        cyclic=cyclic,
        num_workers=num_workers,
    )

    return DataLoader(dataset, batch_size=None)
def _gather_paths(base_dir, patterns):
    """Collect files in base_dir matching patterns recursively."""
    collected = []
    for pattern in patterns:
        collected.extend(Path(base_dir).rglob(pattern))

    unique = []
    seen = set()
    for path in collected:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)

    unique.sort()
    return unique

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

    def forward(
        self,
        white_indices,
        white_offsets,
        black_indices,
        black_offsets,
        white_weights=None,
        black_weights=None,
    ):
        white_transformed = self.activation(
            self.ft_white(white_indices, white_offsets, per_sample_weights=white_weights)
        )
        black_transformed = self.activation(
            self.ft_black(black_indices, black_offsets, per_sample_weights=black_weights)
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
        batch_start_time = time.time()

        prepared = prepare_sparse_batch(batch, device)

        optimizer.zero_grad()
        predictions = model(
            prepared.white_indices,
            prepared.white_offsets,
            prepared.black_indices,
            prepared.black_offsets,
            prepared.white_weights,
            prepared.black_weights,
        )

        loss = compute_loss(
            predictions,
            prepared.scores,
            prepared.outcomes,
            loss_params,
            epoch_progress,
        )

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
        batch_size = prepared.size
        samples_per_sec = batch_size / batch_time if batch_time > 0 else 0

        wandb.log(
            {
                "batch_loss": loss.item(),
                "batch": batch_idx,
                "batch_time_sec": batch_time,
                "samples_per_sec": samples_per_sec,
            }
        )

        if num_batches % 100 == 0 or batch_idx % 500 == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            avg_batch_time = np.mean(batch_times[-100:]) if batch_times else 0.0
            if is_streaming:
                print(
                    "  Batch {}/{} (processed {:.0f}), Avg Loss: {:.6f}, Current Loss: {:.6f}, Batch Time: {:.3f}s, Samples/sec: {:.0f}".format(
                        batch_idx,
                        num_batches,
                        num_batches,
                        avg_loss,
                        loss.item(),
                        batch_time,
                        samples_per_sec,
                    ),
                    flush=True
                )
            else:
                print(
                    "  Batch {}/{} Loss: {:.6f}, Batch Time: {:.3f}s, Samples/sec: {:.0f}".format(
                        batch_idx,
                        len(dataloader),
                        loss.item(),
                        batch_time,
                        samples_per_sec,
                    ),
                    flush=True
                )

    epoch_duration = time.time() - epoch_start_time
    wandb.log({"epoch_time_sec": epoch_duration})
    
    # Print final epoch summary
    print(f"\nEpoch complete: {num_batches} batches processed")

    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, loss_params: LossParams, device, epoch_idx: int, total_epochs: int):
    """Validate the model using the same blended loss as training."""

    model.eval()
    total_loss = 0.0
    num_batches = 0
    epoch_progress = epoch_idx / max(1, total_epochs - 1)

    with torch.no_grad():
        for batch in dataloader:
            prepared = prepare_sparse_batch(batch, device)

            predictions = model(
                prepared.white_indices,
                prepared.white_offsets,
                prepared.black_indices,
                prepared.black_offsets,
                prepared.white_weights,
                prepared.black_weights,
            )
            loss = compute_loss(
                predictions,
                prepared.scores,
                prepared.outcomes,
                loss_params,
                epoch_progress,
            )

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train HalfKP chess neural network')
    parser.add_argument('data_dir', nargs='?', help='Directory containing binpack training data')
    args = parser.parse_args()
    
    # Hyperparameters
    BATCH_SIZE = 16384
    LEARNING_RATE = 8.75e-4
    NUM_EPOCHS = 600
    GAMMA = 0.992
    NUM_WORKERS = 4
    THREADS = 2
    NETWORK_SAVE_PERIOD = 10
    START_LAMBDA = 1.0
    END_LAMBDA = 0.75
    RANDOM_FEN_SKIPPING = 3
    
    # Device - prioritize Metal (MPS) for Mac, then CUDA, then CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using device: CUDA")
    elif torch.backends.mps.is_available():
        mps_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
        if mps_env == "1":
            device = torch.device('mps')
            print("Using device: MPS with CPU fallback for unsupported operators")
        else:
            print(
                "MPS detected but embedding_bag is unsupported; falling back to CPU. "
                "Set PYTORCH_ENABLE_MPS_FALLBACK=1 before launching to enable automatic CPU fallback."
            )
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("Using device: CPU")
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
    
    # Validate binpack files
    print("Validating binpack files...")
    min_file_size = 1024  # Minimum reasonable size
    for binpack_file in binpack_paths:
        if not binpack_file.is_file():
            raise ValueError(f"Binpack path is not a file: {binpack_file}")
        
        file_size = binpack_file.stat().st_size
        if file_size == 0:
            raise ValueError(f"Binpack file is empty: {binpack_file}")
        
        if file_size < min_file_size:
            print(f"  Warning: {binpack_file} is suspiciously small ({file_size} bytes)")
    
    print(f"Validated {len(binpack_paths)} binpack file(s)")

    # Randomly select validation files (~5%)
    binpack_paths_shuffled = binpack_paths.copy()
    np.random.shuffle(binpack_paths_shuffled)
    val_file_count = max(1, len(binpack_paths) // 20)
    val_files = binpack_paths_shuffled[:val_file_count]
    print(f"Using {len(binpack_paths) - len(val_files)} file(s) for training set")
    print(f"Using {len(val_files)} file(s) for validation set")

    train_skip_config = DataloaderSkipConfig(
        filtered=True,
        random_fen_skipping=RANDOM_FEN_SKIPPING,
        wld_filtered=True,
    )
    val_skip_config = DataloaderSkipConfig()

    val_loader = create_sparse_dataloader(
        val_files,
        batch_size=BATCH_SIZE,
        skip_config=val_skip_config,
        num_workers=NUM_WORKERS,
        cyclic=False,
    )

    print("\nCreating sparse training dataloader backed by C++ reader...")
    train_loader = create_sparse_dataloader(
        binpack_paths,
        batch_size=BATCH_SIZE,
        skip_config=train_skip_config,
        num_workers=NUM_WORKERS,
        cyclic=False,
    )
    
    # Initialize model
    model = HalfKPNetwork().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_params = LossParams(start_lambda=START_LAMBDA, end_lambda=END_LAMBDA)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=GAMMA)
    
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
            "start_lambda": START_LAMBDA,
            "end_lambda": END_LAMBDA,
            "random_fen_skipping": RANDOM_FEN_SKIPPING,
            "loader": "nnue_cpp_sparse",
            "device": str(device),
            "num_binpack_files": len(binpack_paths),
            "data_dir": str(data_dir),
        }
    )
    
    # Training loop
    print("\nStarting training with streaming dataset...")
    print("Note: Training on full dataset across all binpack files")
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*60}")
        
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
        val_loss = validate(
            model,
            val_loader,
            loss_params,
            device,
            epoch,
            NUM_EPOCHS,
        )
        
        print(f"\nEpoch {epoch+1} Summary - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
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
