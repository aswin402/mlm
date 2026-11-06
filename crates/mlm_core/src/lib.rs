//! Native Rust GGUF v3 header/metadata/tensor validator and sequence token packer for MLM v2.

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
    pub architecture: String,
    pub quantization_bits: u32,
    pub expert_count: u32,
    pub graph_edge_rank: u32,
    pub tensors_aligned: bool,
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string(r: &mut impl Read) -> io::Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn skip_value(r: &mut impl Read, val_type: u32) -> io::Result<()> {
    match val_type {
        4 | 5 | 6 => {
            let mut b = [0u8; 4];
            r.read_exact(&mut b)
        }
        7 => {
            let mut b = [0u8; 1];
            r.read_exact(&mut b)
        }
        8 => {
            let _ = read_string(r)?;
            Ok(())
        }
        10 => {
            let mut b = [0u8; 8];
            r.read_exact(&mut b)
        }
        9 => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_value(r, elem_type)?;
            }
            Ok(())
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported GGUF value type {val_type}"),
        )),
    }
}

/// Inspect and validate the binary header, metadata KVs, and tensor descriptors of a GGUF v3 file.
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

    let mut architecture = String::from("unknown");
    let mut quantization_bits: u32 = 32;
    let mut expert_count: u32 = 0;
    let mut graph_edge_rank: u32 = 0;

    for _ in 0..metadata_kv_count {
        let key = read_string(&mut file)?;
        let val_type = read_u32(&mut file)?;
        if key == "general.architecture" && val_type == 8 {
            architecture = read_string(&mut file)?;
        } else if key == "general.quantization_bits" && val_type == 4 {
            quantization_bits = read_u32(&mut file)?;
        } else if key.ends_with(".expert_count") && val_type == 4 {
            expert_count = read_u32(&mut file)?;
        } else if key == "spidernet.graph_edge_rank" && val_type == 4 {
            graph_edge_rank = read_u32(&mut file)?;
        } else {
            skip_value(&mut file, val_type)?;
        }
    }

    let mut tensors_aligned = true;
    for _ in 0..tensor_count {
        let _name = read_string(&mut file)?;
        let n_dims = read_u32(&mut file)?;
        for _ in 0..n_dims {
            let _dim = read_u64(&mut file)?;
        }
        let _ttype = read_u32(&mut file)?;
        let offset = read_u64(&mut file)?;
        if offset % 32 != 0 {
            tensors_aligned = false;
        }
    }

    Ok(GgufSummary {
        version,
        tensor_count,
        metadata_kv_count,
        file_size_bytes,
        architecture,
        quantization_bits,
        expert_count,
        graph_edge_rank,
        tensors_aligned,
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
