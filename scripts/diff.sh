#!/bin/bash

# Dump the working diff into changes.log: tracked changes plus
# untracked files as add-only diffs

# tracked files excluded from the dump
EXCLUDES=(
  Cargo.lock
)

source "$(dirname "$0")/common.sh"
setup_root

print_pretty "Collect diff..." "info"

excludes=()
for file in "${EXCLUDES[@]}"; do
  excludes+=(":(exclude)$file")
done

> changes.log
git --no-pager diff -- "${excludes[@]}" | grep -v '^index ' >> changes.log
while IFS= read -r file; do
  git diff --no-index -- /dev/null "$file" | grep -v '^index ' >> changes.log || true
done < <(git ls-files --others --exclude-standard)

print_pretty "Done: changes.log" "success"
