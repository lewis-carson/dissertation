import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import chess
import numpy as np
import pandas as pd
from pathlib import Path
import glob
import wandb


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
        if piece.piece_type == chess.KING:
            # Enemy king (only one piece type for king)
            piece_idx = 5 if piece.color == chess.WHITE else 9
        else:
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
        # Normalize win_prob - values can be extremely small or large
        # Use log-space and clip to reasonable range
        raw_eval = float(row['win_prob'])
        
        # Handle edge cases
        if raw_eval <= 0 or not np.isfinite(raw_eval):
            eval_score = 0.0
        elif raw_eval >= 1.0:
            eval_score = 1.0
        else:
            # Use sigmoid-like transformation to map to [-1, 1]
            # Take log and clip to avoid extreme values
            log_eval = np.log(raw_eval)
            eval_score = np.tanh(log_eval / 10.0)  # Scale and squash
        
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
    """Streaming dataset that loads CSV files one at a time."""
    
    def __init__(self, data_dir, chunk_size=10000, shuffle_files=True):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.shuffle_files = shuffle_files
        
        # Find all CSV files
        self.csv_files = sorted(glob.glob(str(self.data_dir / "*.csv")))
        
        if not self.csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")
        
        print(f"Found {len(self.csv_files)} CSV files to process")
        
    def __iter__(self):
        # Optionally shuffle file order each epoch
        files_to_process = self.csv_files.copy()
        if self.shuffle_files:
            np.random.shuffle(files_to_process)
        
        # Process each file
        for file_idx, csv_file in enumerate(files_to_process):
            if file_idx % 10 == 0:
                print(f"Processing file {file_idx + 1}/{len(files_to_process)}: {Path(csv_file).name}")
            
            try:
                # Read file in chunks to avoid memory issues with very large files
                for chunk in pd.read_csv(csv_file, chunksize=self.chunk_size):
                    # Process each row in the chunk
                    for _, row in chunk.iterrows():
                        result = process_position(row)
                        if result is not None:
                            yield result
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue


class ChessDataset(Dataset):
    """In-memory dataset for validation set (smaller subset)."""
    
    def __init__(self, csv_files, max_samples=10000):
        self.data = []
        
        print(f"Loading validation data from {len(csv_files)} file(s)...")
        
        total_loaded = 0
        for csv_file in csv_files:
            if total_loaded >= max_samples:
                break
                
            try:
                # Read file in chunks
                chunk_size = min(10000, max_samples - total_loaded)
                for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
                    for _, row in chunk.iterrows():
                        if total_loaded >= max_samples:
                            break
                        
                        result = process_position(row)
                        if result is not None:
                            self.data.append(result)
                            total_loaded += 1
                    
                    if total_loaded >= max_samples:
                        break
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.data)} validation positions")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Custom collate function to create sparse tensors."""
    batch_size = len(batch)
    
    # Create sparse feature tensors
    white_features_list = []
    black_features_list = []
    evals = []
    stm_list = []
    
    for item in batch:
        white_features_list.append(item['white_features'])
        black_features_list.append(item['black_features'])
        evals.append(item['eval'])
        stm_list.append(1.0 if item['stm'] == chess.WHITE else -1.0)
    
    # Convert to dense tensors (one-hot encoding)
    white_dense = torch.zeros((batch_size, HALFKP_FEATURES))
    black_dense = torch.zeros((batch_size, HALFKP_FEATURES))
    
    for i, (wf, bf) in enumerate(zip(white_features_list, black_features_list)):
        if wf:
            white_dense[i, wf] = 1.0
        if bf:
            black_dense[i, bf] = 1.0
    
    evals_tensor = torch.tensor(evals, dtype=torch.float32)
    stm_tensor = torch.tensor(stm_list, dtype=torch.float32)
    
    return white_dense, black_dense, evals_tensor, stm_tensor


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
    
    for batch_idx, (white_feat, black_feat, evals, stm) in enumerate(dataloader):
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
        
        # Log batch loss to wandb
        wandb.log({"batch_loss": loss.item(), "batch": batch_idx})
        
        if batch_idx % 500 == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            if is_streaming:
                print(f"  Batch {batch_idx}, Avg Loss: {avg_loss:.6f}, Current Loss: {loss.item():.6f}")
            else:
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for white_feat, black_feat, evals, stm in dataloader:
            white_feat = white_feat.to(device)
            black_feat = black_feat.to(device)
            evals = evals.to(device)
            
            predictions = model(white_feat, black_feat)
            loss = criterion(predictions, evals)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
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
    
    # Data directory
    data_dir = Path(__file__).parent.parent / "data" / "out"
    
    if not data_dir.exists():
        raise ValueError(f"Data directory not found: {data_dir}")
    
    # Get list of all CSV files
    csv_files = sorted(glob.glob(str(data_dir / "*.csv")))
    print(f"Found {len(csv_files)} CSV files in {data_dir}")
    
    if not csv_files:
        raise ValueError(f"No CSV files found in {data_dir}")
    
    # Create validation set from first file(s)
    # Use a small subset for validation to keep it in memory
    val_files = csv_files[:max(1, len(csv_files) // 20)]  # Use ~5% of files for validation
    print(f"Using {len(val_files)} file(s) for validation set")
    
    val_dataset = ChessDataset(val_files, max_samples=VAL_SAMPLES)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)
    
    # Create streaming training dataset (all files)
    print("\nCreating streaming training dataset...")
    train_dataset = StreamingChessDataset(data_dir, chunk_size=10000, shuffle_files=True)
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
            "num_csv_files": len(csv_files),
        }
    )
    
    # Training loop
    print("\nStarting training with streaming dataset...")
    print("Note: Training on full dataset across all CSV files")
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
