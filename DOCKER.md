# Docker Setup for Andros Segmentation

This project uses Docker to provide a reproducible environment with GPU support.

## Prerequisites

1.  **Docker** and **Docker Compose** installed.
2.  **NVIDIA Container Toolkit** installed for GPU support.

## Configuration

1.  Copy `.env.example` to `.env`:
    ```bash
    cp .env.example .env
    ```
2.  Edit `.env` and set `DATASET_DIR` to the absolute path of your dataset on the host machine.

## Usage

### Training
Run the training service (default):
```bash
docker-compose up --build andros-segmentation
```

### Evaluation
Run evaluation:
```bash
docker-compose run --rm evaluate
```

### Generate Masks
Generate masks using trained models:
```bash
docker-compose run --rm generate-masks
```

## Structure
- `Dockerfile`: Single efficient build for GPU usage.
- `docker-compose.yml`: Service definitions using environment variables.
- `checkpoints/`, `outputs/`, `config/`: Mounted volumes for persistence.
