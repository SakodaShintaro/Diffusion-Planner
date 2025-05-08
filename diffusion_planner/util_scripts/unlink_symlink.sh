#!/bin/bash
set -eux

target_dir=$(readlink -f $1)

set +x
target_links=$(find $target_dir -type l)
set -x

for link in $target_links; do
    unlink $link
done
