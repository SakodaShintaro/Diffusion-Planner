#!/bin/bash
set -eux

source_root_dir=$(readlink -f $1)

set +x
files=$(find $source_root_dir -name "*.db")
set -x

for file in $files; do
  echo $file
  ln -sf $file ./trainval
done
