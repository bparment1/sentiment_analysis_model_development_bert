#!/bin/bash
set -e

exec uv run python train.py \
    --tracking_uri "${TRACKING_URI:-file:./mlruns}" \
    --artifact_location "${ARTIFACT_LOCATION:-gs://my-bucket/mlflow/artifacts}" \
    "$@"
