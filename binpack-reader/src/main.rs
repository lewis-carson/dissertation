use sfbinpack::CompressedTrainingDataEntryReader;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use clap::Parser;
use csv::Writer;
use anyhow::Result;
use serde::Serialize;
use std::fs;
use std::thread;
use crossbeam_channel as channel;

#[derive(Parser, Debug)]
#[command(name = "binpack-to-csv")]
#[command(about = "Convert binpack training data files to CSV format", long_about = None)]
struct Args {
    /// Path to input directory
    #[arg(value_name = "INPUT")]
    input_dir: PathBuf,

    /// Path to output directory
    #[arg(value_name = "OUTPUT")]
    output_dir: PathBuf,

    /// File pattern to match binpack files
    #[arg(short = 'f', long, default_value = "*.no-db.binpack")]
    pattern: String,

    /// Print progress every N files
    #[arg(short = 'v', long, default_value = "100000")]
    progress: usize,

    /// Force overwrite existing CSV files
    #[arg(long)]
    force: bool,

    /// Number of worker threads (defaults to number of CPU cores)
    #[arg(short = 'j', long)]
    jobs: Option<usize>,

    /// Stream entries to stdout instead of writing CSV files
    #[arg(long)]
    stdout: bool,

    /// Process only this binpack file when streaming
    #[arg(long, value_name = "FILE")]
    single_file: Option<PathBuf>,
}

#[derive(Serialize)]
struct CsvEntry {
    fen: String,
    uci_move: String,
    score: i16,
    ply: u16,
    result: i16,
}

fn convert_binpack_to_csv(
    input_file: &Path,
    output_file: &Path,
    progress: usize,
) -> Result<usize> {
    // Open input file
    let file = File::open(&input_file)
        .map_err(|e| anyhow::anyhow!("Failed to open input file {}: {}", input_file.display(), e))?;
    let mut reader = CompressedTrainingDataEntryReader::new(file)
        .map_err(|e| anyhow::anyhow!("Failed to create binpack reader: {}", e))?;

    // Create CSV writer
    let csv_file = File::create(&output_file)
        .map_err(|e| anyhow::anyhow!("Failed to create output CSV file {}: {}", output_file.display(), e))?;
    let mut wtr = Writer::from_writer(csv_file);

    let mut count = 0;
    let start_time = std::time::Instant::now();

    // Process entries
    while reader.has_next() {
        let entry = reader.next();

        let fen = entry.pos.fen()
            .map_err(|e| anyhow::anyhow!("Failed to generate FEN: {:?}", e))?;

        let csv_entry = CsvEntry {
            fen,
            uci_move: format!("{:?}", entry.mv.as_uci()),
            score: entry.score,
            ply: entry.ply,
            result: entry.result as i16,
        };

        wtr.serialize(&csv_entry)
            .map_err(|e| anyhow::anyhow!("Failed to write CSV entry: {}", e))?;

        count += 1;

        // Print progress
        if progress > 0 && count % progress == 0 {
            let elapsed = start_time.elapsed();
            let rate = count as f64 / elapsed.as_secs_f64();
            eprintln!(
                "  [{:?}] Processed {} entries ({:.0} entries/sec)...",
                input_file.file_name().unwrap_or_default(),
                count,
                rate
            );
        }
    }

    wtr.flush()
        .map_err(|e| anyhow::anyhow!("Failed to flush CSV writer: {}", e))?;

    let elapsed = start_time.elapsed();
    eprintln!(
        "  ✓ {} → {} ({} entries, {:.2}s)",
        input_file.file_name().unwrap_or_default().to_string_lossy(),
        output_file.file_name().unwrap_or_default().to_string_lossy(),
        count,
        elapsed.as_secs_f64()
    );

    Ok(count)
}

fn stream_binpack_file(input_file: &Path, progress: usize) -> Result<usize> {
    let file = File::open(&input_file)
        .map_err(|e| anyhow::anyhow!("Failed to open input file {}: {}", input_file.display(), e))?;
    let mut reader = CompressedTrainingDataEntryReader::new(file)
        .map_err(|e| anyhow::anyhow!("Failed to create binpack reader: {}", e))?;

    let stdout = io::stdout();
    let mut handle = stdout.lock();

    let mut count = 0usize;
    let start_time = std::time::Instant::now();

    while reader.has_next() {
        let entry = reader.next();

        let fen = entry.pos.fen()
            .map_err(|e| anyhow::anyhow!("Failed to generate FEN: {:?}", e))?;

        if let Err(err) = writeln!(handle, "{}\t{}", fen, entry.score) {
            if err.kind() == io::ErrorKind::BrokenPipe {
                return Ok(count);
            }
            return Err(anyhow::anyhow!("Failed to write to stdout: {}", err));
        }

        count += 1;

        // if progress > 0 && count % progress == 0 {
        //     let elapsed = start_time.elapsed();
        //     let rate = count as f64 / elapsed.as_secs_f64();
        //     eprintln!(
        //         "  [{:?}] Streamed {} entries ({:.0} entries/sec)...",
        //         input_file.file_name().unwrap_or_default(),
        //         count,
        //         rate
        //     );
        // }
    }

    if let Err(err) = handle.flush() {
        if err.kind() != io::ErrorKind::BrokenPipe {
            return Err(anyhow::anyhow!("Failed to flush stdout: {}", err));
        }
    }

    let elapsed = start_time.elapsed();
    eprintln!(
        "  ✓ {} ({} entries, {:.2}s)",
        input_file.file_name().unwrap_or_default().to_string_lossy(),
        count,
        elapsed.as_secs_f64()
    );

    Ok(count)
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Validate input directory exists
    if !args.input_dir.is_dir() {
        return Err(anyhow::anyhow!("Input directory does not exist: {}", args.input_dir.display()));
    }

    // Find all binpack files matching the pattern
    let mut binpack_files: Vec<PathBuf> = fs::read_dir(&args.input_dir)
        .map_err(|e| anyhow::anyhow!("Failed to read input directory: {}", e))?
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let path = entry.path();
            if path.is_file() {
                let filename = path.file_name()?.to_string_lossy();
                // Simple pattern matching - check if filename ends with .binpack or matches pattern
                if filename.contains("binpack") {
                    return Some(path);
                }
            }
            None
        })
        .collect();

    if binpack_files.is_empty() {
        return Err(anyhow::anyhow!("No binpack files found in {}", args.input_dir.display()));
    }

    binpack_files.sort();

    eprintln!("Found {} binpack file(s)\n", binpack_files.len());

    if args.stdout {
        if let Some(single) = &args.single_file {
            let input_file = if single.is_absolute() {
                single.clone()
            } else {
                args.input_dir.join(single)
            };

            if !input_file.is_file() {
                return Err(anyhow::anyhow!(
                    "Binpack file does not exist: {}",
                    input_file.display()
                ));
            }

            eprintln!("Streaming {} to stdout...", input_file.display());
            stream_binpack_file(&input_file, args.progress)?;
        } else {
            eprintln!("Streaming {} binpack file(s) via stdout", binpack_files.len());
            for input_file in &binpack_files {
                stream_binpack_file(input_file, args.progress)?;
            }
        }

        return Ok(());
    }

    // Create output directory if it doesn't exist
    fs::create_dir_all(&args.output_dir)
        .map_err(|e| anyhow::anyhow!("Failed to create output directory {}: {}", args.output_dir.display(), e))?;

    let start_time = std::time::Instant::now();
    let mut total_entries = 0;
    let mut successful = 0;
    let mut skipped = 0;

    // Prepare tasks (skip existing outputs unless force)
    let mut tasks: Vec<(usize, PathBuf, PathBuf)> = Vec::new();
    for (i, input_file) in binpack_files.iter().enumerate() {
        let output_filename = input_file
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string() + ".csv";
        let output_file = args.output_dir.join(&output_filename);

        if output_file.exists() && !args.force {
            skipped += 1;
            eprintln!("[{}/{}] Skipping (already exists)...", i + 1, binpack_files.len());
            eprintln!("  → {}", output_file.file_name().unwrap_or_default().to_string_lossy());
            continue;
        }

        tasks.push((i + 1, input_file.clone(), output_file));
    }

    if tasks.is_empty() {
        eprintln!("No files to process (all outputs exist or no binpack files found)");
        let total_elapsed = start_time.elapsed();
        eprintln!("\n✓ Conversion complete!\n  {} of {} files converted\n  {} total entries\n  {:.2}s total time",
            successful,
            binpack_files.len(),
            total_entries,
            total_elapsed.as_secs_f64()
        );
        return Ok(());
    }

    // Decide number of worker threads
    let jobs = args.jobs.unwrap_or_else(|| num_cpus::get());
    let jobs = std::cmp::min(jobs, tasks.len());

    eprintln!("Processing {} file(s) with {} worker(s) ({} skipped)", tasks.len(), jobs, skipped);

    // Create channels for tasks and results
    let (task_s, task_r) = channel::unbounded::<(usize, PathBuf, PathBuf)>();
    let (res_s, res_r) = channel::unbounded::<(usize, PathBuf, PathBuf, Result<usize, String>)>();

    // Spawn worker threads
    for worker_id in 0..jobs {
        let task_r = task_r.clone();
        let res_s = res_s.clone();
        let progress = args.progress;

        thread::spawn(move || {
            while let Ok((idx, input, output)) = task_r.recv() {
                eprintln!("[worker-{}][{}/?] Converting {} -> {}...", worker_id + 1, idx, input.display(), output.display());
                let res = match convert_binpack_to_csv(&input, &output, progress) {
                    Ok(count) => Ok(count),
                    Err(e) => Err(format!("{}", e)),
                };

                // send result back
                let _ = res_s.send((idx, input, output, res));
            }
        });
    }

    // Send tasks
    for t in tasks.into_iter() {
        let _ = task_s.send(t);
    }
    // Drop sender so workers can exit when done
    drop(task_s);

    // Collect results
    let mut received = 0usize;
    let expected = binpack_files.len() - skipped;
    while let Ok((idx, input, output, res)) = res_r.recv() {
        received += 1;
        match res {
            Ok(count) => {
                total_entries += count;
                successful += 1;
                eprintln!("[{}] OK: {} entries from {} -> {}", idx, count, input.display(), output.display());
            }
            Err(errstr) => {
                eprintln!("[{}] ✗ Error converting {}: {}", idx, input.display(), errstr);
            }
        }

        if received >= expected {
            break;
        }
    }

    let total_elapsed = start_time.elapsed();
    eprintln!(
        "\n✓ Conversion complete!\n  {} of {} files converted\n  {} total entries\n  {:.2}s total time",
        successful,
        binpack_files.len(),
        total_entries,
        total_elapsed.as_secs_f64()
    );

    Ok(())
}