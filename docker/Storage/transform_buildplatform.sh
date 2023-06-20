#!/bin/sh

if [ "$1" = "arm64" ]; then
    echo "aarch64"
else
    echo "x86_64"
fi