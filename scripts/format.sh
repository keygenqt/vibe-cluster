#!/bin/bash

source "$(dirname "$0")/common.sh"
setup_root

# Format Rust
print_pretty "Format Rust..." "info"
rustup run nightly cargo fmt

print_pretty "Done" "success"
