FROM python:3.12-slim

WORKDIR /app
COPY src ./src
RUN pip install --no-cache-dir \
    "fastapi>=0.115,<1" \
    "httpx>=0.28,<1" \
    "mlflow==3.14.0" \
    "pyyaml>=6,<7" \
    "uvicorn[standard]>=0.34,<1"
ENV PYTHONPATH=/app/src

EXPOSE 8000
CMD ["uvicorn", "strata_ot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
