#!/bin/bash
set -e

exec uv run python train.py \
    --tracking_uri "${TRACKING_URI:-file:./mlruns}" \
    ${ARTIFACT_LOCATION:+--artifact_location "$ARTIFACT_LOCATION"} \
    "$@"
