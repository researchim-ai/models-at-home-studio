# Backend container — extends the base models-at-home image with FastAPI.
# Electron starts this final image, not the raw Streamlit/ML base image.
ARG BASE_IMAGE=models-at-home-base:latest
FROM ${BASE_IMAGE}

ARG STUDIO_BUILD_ID=manual
LABEL com.modelsathome.studio.build=${STUDIO_BUILD_ID}
ENV MODELS_AT_HOME_ROOT=/app

# Install the API and supervisor dependencies used by this source tree.
COPY backend/requirements.txt /tmp/mah-requirements.txt
# uv is not guaranteed to survive the ML image's multi-stage build.
COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /usr/local/bin/uv
RUN uv pip install --python /opt/venv/bin/python -r /tmp/mah-requirements.txt

# Keep the packaged native algorithms in sync with the Studio bridge.
COPY models-at-home/homellm/ /app/homellm/
COPY models-at-home/LICENSE /app/LICENSE.homellm

# Copy backend API code
COPY backend/ /app/backend/
COPY examples/ /app/examples/
COPY requirements-world-foundation.txt /app/requirements-world-foundation.txt
COPY requirements-world-local.txt /app/requirements-world-local.txt

EXPOSE 8000 8888

CMD ["python", "-m", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
