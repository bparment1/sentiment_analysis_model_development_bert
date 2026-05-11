ROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies from lockfile (separate layer for caching)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source code
COPY train.py .
COPY sentiment_model/ sentiment_model/
COPY configs/ configs/
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Stream logs immediately
ENV PYTHONUNBUFFERED=1

# Override at runtime: docker run -e TRACKING_URI=... -e ARTIFACT_LOCATION=... <image>
ENV TRACKING_URI=file:./mlruns
ENV ARTIFACT_LOCATION=gs://my-bucket/mlflow/artifacts

ENTRYPOINT ["./entrypoint.sh"]

# Default args — override by passing args after the image name:
#   docker run <image> --model_name bert-base-uncased --freeze_base --number_epoch 3
CMD ["--model_name", "bert-base-uncased", \
     "--number_epoch", "1", \
     "--experiment_name", "bert-train-sentiment"]
