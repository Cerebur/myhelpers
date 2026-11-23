#!/bin/zsh

set -eu

if (( $# != 2 )); then
  print -u2 "Usage: $0 SOURCE DESTINATION"
  exit 2
fi

source_path=$1
destination_path=$2

if [[ ! -f "$source_path" ]]; then
  print -u2 "Source file does not exist: $source_path"
  exit 1
fi

source_size=$(wc -c < "$source_path" | tr -d '[:space:]')
if [[ "$source_size" == "0" ]]; then
  print -u2 "Source file is empty: $source_path"
  exit 1
fi

destination_directory=${destination_path:h}
destination_name=${destination_path:t}
mkdir -p "$destination_directory"

intermediate_path="$destination_directory/.${destination_name}.part.$$"
cleanup() {
  [[ -e "$intermediate_path" ]] && rm -f "$intermediate_path"
}
trap cleanup EXIT HUP INT TERM

if ! rsync -- "$source_path" "$intermediate_path"; then
  print -u2 "rsync failed copying $source_path to $intermediate_path"
  exit 1
fi

if [[ ! -f "$intermediate_path" ]]; then
  print -u2 "rsync did not create the intermediate file: $intermediate_path"
  exit 1
fi

copied_size=$(wc -c < "$intermediate_path" | tr -d '[:space:]')
if [[ "$copied_size" != "$source_size" ]]; then
  print -u2 "Copied size mismatch for $destination_path: expected $source_size, got $copied_size"
  exit 1
fi

mv -f "$intermediate_path" "$destination_path"
trap - EXIT HUP INT TERM
