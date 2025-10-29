use sfbinpack::CompressedTrainingDataEntryReader;
use std::fs::File;

fn main() {
    let file = File::open("../data-test80/test80-jan2024/training-run1-test80-20240101-0017.no-db.binpack").unwrap();
    let mut reader = CompressedTrainingDataEntryReader::new(file).unwrap();

    while reader.has_next() {
        let entry = reader.next();

        println!("entry:");
        println!("fen {}", entry.pos.fen().unwrap());
        println!("uci {:?}", entry.mv.as_uci());
        println!("score {}", entry.score);
        println!("ply {}", entry.ply);
        println!("result {}", entry.result);
        println!("\n");
    }
}