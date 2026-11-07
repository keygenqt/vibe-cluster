#!/bin/bash

# Setup root directory
setup_root() {
  ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
  cd "$ROOT_DIR" || exit 1
}

# Print styled status message
# $1 - text, $2 - level: info | success | error
print_pretty() {
  local color
  case "$2" in
  info) color="\033[0;34m" ;;
  success) color="\033[0;32m" ;;
  error) color="\033[0;31m" ;;
  *) color="\033[0m" ;;
  esac
  echo -e "${color}$1\033[0m"
}
