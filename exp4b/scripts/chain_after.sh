#!/bin/bash
cd "$(dirname "$0")/.."
python scripts/batch_seq2.py > reports/batch_seq2.log 2>&1
python scripts/leak_seq.py > reports/leak_seq.log 2>&1
