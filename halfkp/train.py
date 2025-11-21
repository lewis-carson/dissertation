import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import threading
import queue

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.amp import GradScaler, autocast
import wandb

import binpack_loader


# HalfKP feature dimensions (matching nnue-pytorch reference implementation)
NUM_SQUARES = 64
NUM_PIECE_TYPES = 10  # 5 piece types (P, N, B, R, Q) * 2 colours, kings excluded
NUM_PLANES = NUM_SQUARES * NUM_PIECE_TYPES + 1  # extra plane for bias bucket
HALFKP_FEATURES = NUM_PLANES * NUM_SQUARES  # 64 * (64 * 10 + 1) = 41_024


def _get_env_value(name: str, default, cast_func):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return cast_func(raw)
    except (TypeError, ValueError):
        print(f"Ignoring invalid value for {name}: {raw}")
        return default


def _get_env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def orient(is_white_pov: bool, square: int) -> int:
    """Mirror board for black perspective to share weights."""
    return (63 * (not is_white_pov)) ^ square


def halfkp_index(is_white_pov: bool, king_bucket: int, square: int, piece: chess.Piece) -> int:
    """Compute HalfKP feature index, aligned with nnue-pytorch layout."""
    piece_index = (piece.piece_type - 1) * 2 + int(piece.color != is_white_pov)
    return 1 + orient(is_white_pov, square) + piece_index * NUM_SQUARES + king_bucket * NUM_PLANES


@dataclass
class DataloaderSkipConfig:
    filtered: bool = False
    random_fen_skipping: int = 0
    wld_filtered: bool = False
    early_fen_skipping: int = -1
    simple_eval_skipping: int = -1
    param_index: int = 0

    def to_dict(self) -> dict:
        return {
            "filtered": self.filtered,
            "random_fen_skipping": self.random_fen_skipping,
            "wld_filtered": self.wld_filtered,
            "early_fen_skipping": self.early_fen_skipping,
            "simple_eval_skipping": self.simple_eval_skipping,
            "param_index": self.param_index,
        }


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


def _tensor_from_numpy(array: np.ndarray, device: torch.device, *, dtype=None) -> torch.Tensor:
    tensor = torch.from_numpy(array)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)

    if device.type == "cuda":
        return tensor.pin_memory().to(device=device, non_blocking=True)
    if device.type == "cpu":
        return tensor.clone()
    return tensor.to(device=device)


def _convert_numpy_batch_to_tensors(batch, device: torch.device):
    (
        us_np,
        them_np,
        white_idx_np,
        white_vals_np,
        black_idx_np,
        black_vals_np,
        outcome_np,
        score_np,
        psqt_idx_np,
        layer_stack_np,
    ) = batch

    us = _tensor_from_numpy(us_np, device, dtype=torch.float32)
    them = _tensor_from_numpy(them_np, device, dtype=torch.float32)
    white_indices = _tensor_from_numpy(white_idx_np, device, dtype=torch.long)
    white_values = _tensor_from_numpy(white_vals_np, device, dtype=torch.float32)
    black_indices = _tensor_from_numpy(black_idx_np, device, dtype=torch.long)
    black_values = _tensor_from_numpy(black_vals_np, device, dtype=torch.float32)
    outcome = _tensor_from_numpy(outcome_np, device, dtype=torch.float32)
    score = _tensor_from_numpy(score_np, device, dtype=torch.float32)
    psqt_indices = _tensor_from_numpy(psqt_idx_np, device, dtype=torch.long)
    layer_stack_indices = _tensor_from_numpy(layer_stack_np, device, dtype=torch.long)

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


class SparseBatchIterableDataset(IterableDataset):
    def __init__(
        self,
        feature_set: str,
        filenames,
        batch_size: int,
        skip_config: DataloaderSkipConfig,
        cyclic: bool,
        num_workers: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.feature_set = feature_set
        self.filenames = list(filenames)
        self.batch_size = batch_size
        self.skip_config = skip_config
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.device = device or torch.device("cpu")

    def __iter__(self):
        if not self.filenames:
            raise ValueError("No binpack files provided to sparse loader")

        stream = binpack_loader.SparseBatchStream(
            feature_set=self.feature_set,
            files=[str(path) for path in self.filenames],
            batch_size=self.batch_size,
            skip_config=self.skip_config.to_dict(),
            cyclic=self.cyclic,
            num_workers=max(1, self.num_workers),
        )

        device = self.device
        batch_count = 0

        while True:
            try:
                batch = next(stream)
            except StopIteration:
                break
            except Exception as exc:  # pragma: no cover - defensive path
                raise RuntimeError(
                    f"Error fetching batch {batch_count} from Rust dataloader: {exc}"
                ) from exc

            try:
                tensors = _convert_numpy_batch_to_tensors(batch, device)
            except Exception as exc:  # pragma: no cover - defensive path
                raise RuntimeError(
                    f"Error converting batch {batch_count} to tensors: {exc}"
                ) from exc

            batch_count += 1
            yield tensors


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
    device: Optional[torch.device] = None,
    epoch_size: Optional[int] = None,
):
    """Build a DataLoader backed by the Rust sparse pipeline exposed via PyO3."""

    files = [str(Path(path)) for path in binpack_files]
    dataset = SparseBatchIterableDataset(
        "HalfKP",
        files,
        batch_size,
        skip_config=skip_config,
        cyclic=cyclic,
        num_workers=num_workers,
        device=device,
    )

    # Wrap streaming dataset into a fixed number of batches per epoch when epoch_size is provided
    if epoch_size is not None and epoch_size > 0:
        num_batches = (epoch_size + batch_size - 1) // batch_size

        class FixedNumBatchesDataset(torch.utils.data.Dataset):
            def __init__(self, iterable_ds, num_batches):
                super().__init__()
                self.iterable = iterable_ds
                self.num_batches = num_batches
                self._iter = None
                self._prefetch_queue = queue.Queue(maxsize=100)
                self._prefetch_thread = None
                self._stop_event = threading.Event()
                self._started = False
                self._lock = threading.Lock()

            def _prefetch_worker(self):
                try:
                    it = iter(self.iterable)
                    while not self._stop_event.is_set():
                        try:
                            item = next(it)
                        except StopIteration:
                            self._prefetch_queue.put(None)
                            break
                        self._prefetch_queue.put(item)
                except Exception as exc:
                    self._prefetch_queue.put(exc)

            def _start(self):
                with self._lock:
                    if not self._started:
                        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
                        self._prefetch_thread.start()
                        self._started = True

            def __len__(self):
                return self.num_batches

            def __getitem__(self, idx):
                self._start()
                item = self._prefetch_queue.get()
                if item is None:
                    raise StopIteration
                if isinstance(item, Exception):
                    raise item
                return item

            def __del__(self):
                if hasattr(self, "_stop_event"):
                    self._stop_event.set()
                if hasattr(self, "_prefetch_thread") and self._prefetch_thread:
                    self._prefetch_thread.join(timeout=1.0)

        dataset = FixedNumBatchesDataset(dataset, num_batches)

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
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    log_gpu_memory: bool = False,
    scheduler=None,
    scheduler_step_per_optim_step: bool = False,
):
    """Train for one epoch using nnue-pytorch style blended loss."""

    model.train()
    total_loss = 0.0
    num_batches = 0
    epoch_start_time = time.time()
    batch_times = []

    epoch_progress = epoch_idx / max(1, total_epochs - 1)

    scaler = GradScaler(enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)
    grad_accum_steps = max(1, grad_accum_steps)

    def _optimizer_step():
        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # When step-based decay is enabled, advance scheduler on each optimizer step
        if scheduler is not None and scheduler_step_per_optim_step:
            try:
                scheduler.step()
            except Exception:
                pass

    for batch_idx, batch in enumerate(dataloader):
        batch_start_time = time.time()

        prepared = prepare_sparse_batch(batch, device)

        amp_enabled = use_amp and device.type == "cuda"
        amp_device_type = "cuda" if device.type == "cuda" else "cpu"
        with autocast(amp_device_type, enabled=amp_enabled):
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
            optimizer.zero_grad(set_to_none=True)
            continue

        scaled_loss = loss / grad_accum_steps
        if use_amp:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            _optimizer_step()

        total_loss += loss.item()
        num_batches += 1

        batch_time = time.time() - batch_start_time
        batch_times.append(batch_time)
        batch_size = prepared.size
        samples_per_sec = batch_size / batch_time if batch_time > 0 else 0

        log_payload = {
            "batch_loss": loss.item(),
            "batch": batch_idx,
            "batch_time_sec": batch_time,
            "samples_per_sec": samples_per_sec,
        }
        if log_gpu_memory and device.type == "cuda":
            log_payload.update(
                {
                    "gpu_mem_allocated_mb": torch.cuda.memory_allocated(device) / 1024 ** 2,
                    "gpu_mem_reserved_mb": torch.cuda.memory_reserved(device) / 1024 ** 2,
                }
            )
        wandb.log(log_payload)

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
        #print("batch done")

    if num_batches % grad_accum_steps != 0:
        _optimizer_step()

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
    
    # Hyperparameters (env override friendly)
    BATCH_SIZE = _get_env_value("BATCH_SIZE", 16384, int)
    LEARNING_RATE = _get_env_value("LEARNING_RATE", 0.01, float)
    NUM_EPOCHS = _get_env_value("NUM_EPOCHS", 600, int)
    GAMMA = _get_env_value("GAMMA", 0.992, float)
    NUM_WORKERS = max(0, _get_env_value("NUM_WORKERS", 1, int))
    THREADS = max(0, _get_env_value("THREADS", 4, int))
    NETWORK_SAVE_PERIOD = _get_env_value("NETWORK_SAVE_PERIOD", 10, int)
    START_LAMBDA = _get_env_value("START_LAMBDA", 1.0, float)
    END_LAMBDA = _get_env_value("END_LAMBDA", 0.75, float)
    RANDOM_FEN_SKIPPING = max(0, _get_env_value("RANDOM_FEN_SKIPPING", 0, int))
    GRAD_ACCUM_STEPS = max(1, _get_env_value("ACCUM_STEPS", 1, int))
    USE_AMP = _get_env_flag("USE_AMP", True)
    LOG_GPU_MEMORY = _get_env_flag("LOG_GPU_MEMORY", False)
    OPTIMIZER = os.environ.get("OPTIMIZER", "adagrad").lower()
    # Step-based LR decay controls (defaults preserve previous behavior)
    LR_DECAY_BY_STEPS = _get_env_flag("LR_DECAY_BY_STEPS", False)
    LR_DECAY_STEP_SIZE = max(1, _get_env_value("LR_DECAY_STEP_SIZE", 1000, int))
    LR_DECAY_GAMMA = _get_env_value("LR_DECAY_GAMMA", 0.992, float)
    EPOCH_SIZE = _get_env_value("EPOCH_SIZE", 100000000, int)
    VAL_SIZE = _get_env_value("VAL_SIZE", 1000000, int)

    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be positive")
    if NUM_EPOCHS <= 0:
        raise ValueError("NUM_EPOCHS must be positive")
    if NETWORK_SAVE_PERIOD is not None and NETWORK_SAVE_PERIOD <= 0:
        NETWORK_SAVE_PERIOD = None
    
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

    if device.type == 'cuda':
        # torch.cuda.set_device expects an integer or device index; device may be 'cuda' without index
        try:
            cuda_index = device.index if device.index is not None else torch.cuda.current_device()
            torch.cuda.set_device(cuda_index)
        except Exception as exc:  # pragma: no cover - defensive safety
            print(f"Warning: couldn't set cuda device index from device={device}: {exc}")
            # best effort: don't set device

    use_amp = USE_AMP and device.type == 'cuda'
    if USE_AMP and not use_amp:
        print("AMP requested but CUDA is unavailable; running without mixed precision.")

    initial_device_stats = {}
    if device.type == 'cuda' and LOG_GPU_MEMORY:
        cuda_index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(cuda_index)
        initial_device_stats = {
            "gpu_name": props.name,
            "gpu_total_memory_gb": props.total_memory / (1024 ** 3),
        }
        alloc_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
        reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
        print(
            f"GPU memory (visible): {props.total_memory / (1024 ** 3):.2f} GB | "
            f"allocated {alloc_mb:.1f} MB | reserved {reserved_mb:.1f} MB"
        )

    if THREADS > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
        os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
        os.environ.setdefault("NUMEXPR_NUM_THREADS", str(THREADS))
        torch.set_num_threads(THREADS)
        try:
            torch.set_num_interop_threads(max(1, THREADS // 2))
        except RuntimeError:
            pass
    
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
        filtered=False,
        random_fen_skipping=RANDOM_FEN_SKIPPING,
        wld_filtered=False,
    )
    val_skip_config = DataloaderSkipConfig()

    val_loader = create_sparse_dataloader(
        val_files,
        batch_size=BATCH_SIZE,
        skip_config=val_skip_config,
        num_workers=NUM_WORKERS,
        cyclic=False,
        device=device,
        epoch_size=VAL_SIZE,
    )

    print("\nCreating sparse training dataloader backed by C++ reader...")
    train_loader = create_sparse_dataloader(
        binpack_paths,
        batch_size=BATCH_SIZE,
        skip_config=train_skip_config,
        num_workers=NUM_WORKERS,
        cyclic=False,
        device=device,
        epoch_size=EPOCH_SIZE,
    )
    
    # Initialize model
    model = HalfKPNetwork().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_params = LossParams(start_lambda=START_LAMBDA, end_lambda=END_LAMBDA)

    if OPTIMIZER == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == "adam":
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    else:
        print(f"Unknown OPTIMIZER={OPTIMIZER}, falling back to Adam")
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Scheduler: if LR_DECAY_BY_STEPS then decay every LR_DECAY_STEP_SIZE optimizer steps; otherwise per epoch
    if LR_DECAY_BY_STEPS:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=LR_DECAY_STEP_SIZE, gamma=LR_DECAY_GAMMA
        )
    else:
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
            "accum_steps": GRAD_ACCUM_STEPS,
            "use_amp": use_amp,
            "optimizer": OPTIMIZER,
            "log_gpu_memory": LOG_GPU_MEMORY,
        }
    )
    if initial_device_stats:
        wandb.log(initial_device_stats)
    
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
            use_amp=use_amp,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            log_gpu_memory=LOG_GPU_MEMORY,
            scheduler=scheduler,
            scheduler_step_per_optim_step=LR_DECAY_BY_STEPS,
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

        if not LR_DECAY_BY_STEPS and scheduler is not None:
            scheduler.step()
        
        # Save checkpoint
        if NETWORK_SAVE_PERIOD and (epoch + 1) % NETWORK_SAVE_PERIOD == 0:
            checkpoint_path = Path(__file__).parent / f"halfkp_epoch_{epoch+1}.pt"
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            if scheduler is not None:
                save_dict['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(save_dict, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_path = Path(__file__).parent / "halfkp_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
