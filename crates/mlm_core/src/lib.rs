//! Native Rust GGUF v3 header/tensor validator and sequence token packer for MLM v2.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
pub const GGUF_VERSION_V3: u32 = 3;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufSummary {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub file_size_bytes: u64,
}

/// Inspect and validate the binary header of a GGUF v3 model file.
pub fn inspect_gguf_header(path: &Path) -> io::Result<GgufSummary> {
    let mut file = File::open(path)?;
    let file_size_bytes = file.metadata()?.len();
    let mut header = [0u8; 24];
    file.read_exact(&mut header)?;

    if &header[0..4] != GGUF_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid GGUF magic bytes (expected 'GGUF')",
        ));
    }
    let version = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
    if version != GGUF_VERSION_V3 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported GGUF version {version} (expected 3)"),
        ));
    }
    let tensor_count = u64::from_le_bytes(header[8..16].try_into().unwrap());
    let metadata_kv_count = u64::from_le_bytes(header[16..24].try_into().unwrap());

    Ok(GgufSummary {
        version,
        tensor_count,
        metadata_kv_count,
        file_size_bytes,
    })
}

/// Pack variable-length token sequences into fixed-length `seq_len` blocks.
pub fn pack_token_sequences(
    sequences: &[Vec<u32>],
    seq_len: usize,
    pad_id: u32,
) -> Vec<Vec<u32>> {
    if seq_len == 0 {
        return Vec::new();
    }
    let mut blocks = Vec::new();
    let mut current = Vec::with_capacity(seq_len);
    for seq in sequences {
        for &tok in seq {
            current.push(tok);
            if current.len() == seq_len {
                blocks.push(current);
                current = Vec::with_capacity(seq_len);
            }
        }
    }
    if !current.is_empty() {
        current.resize(seq_len, pad_id);
        blocks.push(current);
    }
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_token_sequences() {
        let seqs = vec![vec![0, 10, 11, 1], vec![0, 20, 1]];
        let packed = pack_token_sequences(&seqs, 4, 2);
        assert_eq!(packed.len(), 2);
        assert_eq!(packed[0], vec![0, 10, 11, 1]);
        assert_eq!(packed[1], vec![0, 20, 1, 2]);
    }
}
