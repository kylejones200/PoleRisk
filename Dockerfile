# Multi-stage build for polerisk platform
FROM python:3.10-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libhdf5-dev \
    libnetcdf-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust for building extensions
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set work directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt pyproject.toml setup.py ./
COPY polerisk_rs/ ./polerisk_rs/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Build Rust extensions
WORKDIR /app/polerisk_rs
RUN pip install maturin
RUN maturin build --release
RUN pip install target/wheels/*.whl

# Return to app directory
WORKDIR /app

# Production stage
FROM python:3.10-slim as production

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-103 \
    libnetcdf19 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r polerisk && useradd -r -g polerisk polerisk

# Set work directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY polerisk/ ./polerisk/
COPY *.py ./
COPY README*.md ./

# Create necessary directories
RUN mkdir -p data output uploads logs && \
    chown -R polerisk:polerisk /app

# Switch to non-root user
USER soilmoisture

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=launch_web_app.py
ENV DATA_DIR=/app/data
ENV OUTPUT_DIR=/app/output
ENV UPLOAD_FOLDER=/app/uploads

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose port
EXPOSE 5000

# Default command
CMD ["python", "launch_web_app.py", "--host", "0.0.0.0", "--port", "5000"]
