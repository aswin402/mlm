use std::env;
use std::path::Path;
use std::process;

use mlm_core::inspect_gguf_header;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: mlm-gguf-inspect <path-to-model.gguf>");
        process::exit(2);
    }
    let path = Path::new(&args[1]);
    match inspect_gguf_header(path) {
        Ok(summary) => {
            println!(
                "{{\"valid\":true,\"version\":{},\"tensor_count\":{},\"metadata_kv_count\":{},\"file_size_bytes\":{}}}",
                summary.version,
                summary.tensor_count,
                summary.metadata_kv_count,
                summary.file_size_bytes
            );
        }
        Err(err) => {
            eprintln!("{{\"valid\":false,\"error\":{:?}}}", err.to_string());
            process::exit(1);
        }
    }
}
