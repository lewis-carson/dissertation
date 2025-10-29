#!/usr/bin/env python3
"""Extract chess data from .bag files to CSV format."""

import csv
import io
import mmap
import os
import struct
from pathlib import Path


class BagFileReader:
    """Simple reader for .bag files (non-compressed bagz format)."""
    
    def __init__(self, filename: str):
        """Initialize the bag file reader."""
        self._filename = filename
        fd = os.open(filename, os.O_RDONLY)
        try:
            self._records = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            file_size = self._records.size()
        except ValueError:
            self._records = b''
            file_size = 0
        finally:
            os.close(fd)
        
        if 0 < file_size < 8:
            raise ValueError('Bag file too small')
        
        self._limits = self._records
        if file_size:
            (index_start,) = struct.unpack('<Q', self._records[-8:])
        else:
            index_start = 0
        
        assert file_size >= index_start
        index_size = file_size - index_start
        assert index_size % 8 == 0
        self._num_records = index_size // 8
        self._limits_start = index_start
    
    def __len__(self):
        """Return the number of records in the bag file."""
        return self._num_records
    
    def __getitem__(self, index):
        """Get a record from the bag file."""
        i = index
        if not 0 <= i < self._num_records:
            raise IndexError('BagReader index out of range')
        end = i * 8 + self._limits_start
        if i:
            rec_range = struct.unpack('<2q', self._limits[end - 8 : end + 8])
        else:
            rec_range = (0, *struct.unpack('<q', self._limits[end : end + 8]))
        return self._records[slice(*rec_range)]


def decode_varint(data: io.BytesIO) -> int:
    """Decode a varint from the data stream."""
    result = 0
    shift = 0
    while True:
        byte = data.read(1)
        if not byte:
            raise ValueError("Unexpected end of stream")
        b = byte[0]
        result |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return result


def decode_bytes(data: io.BytesIO) -> bytes:
    """Decode a length-prefixed byte string."""
    length = decode_varint(data)
    return data.read(length)


def decode_string(data: io.BytesIO) -> str:
    """Decode a UTF-8 string."""
    return decode_bytes(data).decode('utf-8')


def decode_float(data: io.BytesIO) -> float:
    """Decode a double-precision float."""
    float_bytes = data.read(8)
    return struct.unpack('<d', float_bytes)[0]


def decode_action_value(record: bytes) -> tuple:
    """Decode an action_value record (fen, move, win_prob)."""
    data = io.BytesIO(record)
    fen = decode_string(data)
    move = decode_string(data)
    win_prob = decode_float(data)
    return (fen, move, win_prob)


def decode_state_value(record: bytes) -> tuple:
    """Decode a state_value record (fen, win_prob)."""
    data = io.BytesIO(record)
    fen = decode_string(data)
    win_prob = decode_float(data)
    return (fen, win_prob)


def decode_behavioral_cloning(record: bytes) -> tuple:
    """Decode a behavioral_cloning record (fen, move)."""
    data = io.BytesIO(record)
    fen = decode_string(data)
    move = decode_string(data)
    return (fen, move)


def extract_to_csv(bag_file: str, output_dir: str, data_type: str = "action_value"):
    """
    Extract data from a .bag file to CSV format.
    
    Args:
        bag_file: Path to the .bag file
        output_dir: Directory to save CSV files
        data_type: Type of data ('action_value', 'state_value', or 'behavioral_cloning')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output filename
    bag_filename = os.path.basename(bag_file)
    csv_filename = bag_filename.replace('.bag', '.csv')
    output_file = os.path.join(output_dir, csv_filename)
    
    print(f"Processing {bag_file}...")
    print(f"Output: {output_file}")
    
    # Get the appropriate decoder for the data type
    if data_type == "action_value":
        decoder = decode_action_value
    elif data_type == "state_value":
        decoder = decode_state_value
    elif data_type == "behavioral_cloning":
        decoder = decode_behavioral_cloning
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    # Open the bag file
    reader = BagFileReader(bag_file)
    
    print(f"Found {len(reader)} records")
    
    # Determine CSV headers based on data type
    if data_type == "action_value":
        headers = ["fen", "move", "win_prob"]
    elif data_type == "state_value":
        headers = ["fen", "win_prob"]
    elif data_type == "behavioral_cloning":
        headers = ["fen", "move"]
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    # Write to CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        # Process records
        for i, record in enumerate(reader):
            if i % 100000 == 0 and i > 0:
                print(f"  Processed {i:,} records...")
            
            try:
                # Decode the record
                decoded = decoder(record)
                
                # Write to CSV
                writer.writerow(decoded)
            except Exception as e:
                print(f"  Warning: Failed to decode record {i}: {e}")
                continue
    
    print(f"✓ Completed! Wrote {len(reader):,} records to {output_file}")


def main():
    """Main extraction function."""
    # Detect all .bag files in data/data directory
    data_dir = Path(__file__).parent / "data"
    output_dir = Path(__file__).parent / "out"
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    bag_files = list(data_dir.glob("*.bag"))
    
    if not bag_files:
        print(f"No .bag files found in {data_dir}")
        return
    
    print(f"Found {len(bag_files)} .bag file(s)")
    print()
    
    for bag_file in sorted(bag_files):
        # Determine data type from filename
        filename = bag_file.name
        if "action_value" in filename:
            data_type = "action_value"
        elif "state_value" in filename:
            data_type = "state_value"
        elif "behavioral_cloning" in filename:
            data_type = "behavioral_cloning"
        else:
            print(f"Warning: Cannot determine data type for {filename}, skipping...")
            continue
        
        try:
            extract_to_csv(str(bag_file), str(output_dir), data_type)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            import traceback
            traceback.print_exc()
        
        print()


if __name__ == "__main__":
    main()
