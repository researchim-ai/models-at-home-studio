# Backend container — extends the base models-at-home image with FastAPI.
# Electron starts this final image, not the raw Streamlit/ML base image.
FROM models-at-home-base:latest

ENV MODELS_AT_HOME_ROOT=/app

# Install FastAPI + uvicorn on top of existing ML dependencies
RUN uv pip install --python /opt/venv/bin/python \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.34.0" \
    "websockets>=14.0"

# Copy backend API code
COPY backend/ /app/backend/

EXPOSE 8000 8888

CMD ["python", "-m", "uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
