#!/bin/bash
set -eu

# This script convert the dataset to a list.
# $ ls -1
# us-nv-las-vegas-strip_150e044bab0a57a8.npz
# us-nv-las-vegas-strip_2324ed8cd5a75cb9.npz
# us-nv-las-vegas-strip_47ceedefd88158a3.npz
# us-nv-las-vegas-strip_4e507af0fcf75ad3.npz
# us-nv-las-vegas-strip_60b40cbd09e85a06.npz
# ...
# ->
# [
#     "/abs_path/to/us-nv-las-vegas-strip_150e044bab0a57a8.npz",
#     "/abs_path/to/us-nv-las-vegas-strip_2324ed8cd5a75cb9.npz",
#     "/abs_path/to/us-nv-las-vegas-strip_47ceedefd88158a3.npz",
#     "/abs_path/to/us-nv-las-vegas-strip_4e507af0fcf75ad3.npz",
#     "/abs_path/to/us-nv-las-vegas-strip_60b40cbd09e85a06.npz",
#     ...
# ]

target_dir=$1
basename=$(basename $target_dir)
output_file="$target_dir/../$basename.json"

body=$(find $target_dir -name "*.npz" | sed 's/^/    "/g' | sed 's/$/"/g' | sed 's/$/,/g' | sed '$ s/.$//')

echo "[" > $output_file
echo "$body" >> $output_file
echo "]" >> $output_file
echo "output_file: $output_file"
