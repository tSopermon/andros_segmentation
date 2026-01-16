.PHONY: help build train evaluate generate shell test clean cpu up down logs rebuild

# Default target
help:
	@echo "Andros Segmentation - Docker Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build      - Build the Docker image"
	@echo "  train      - Run training with GPU support"
	@echo "  evaluate   - Run evaluation on test set"
	@echo "  generate   - Generate segmentation masks"
	@echo "  shell      - Start interactive bash shell"
	@echo "  test       - Run pytest test suite"
	@echo "  cpu        - Run training without GPU"
	@echo "  up         - Start services with docker-compose"
	@echo "  down       - Stop docker-compose services"
	@echo "  logs       - View docker-compose logs"
	@echo "  rebuild    - Rebuild image from scratch"
	@echo "  clean      - Clean up Docker resources"
	@echo ""
	@echo "Examples:"
	@echo "  make build && make train"
	@echo "  make shell"
	@echo "  make cpu"

# Build Docker image
build:
	@echo "Building Docker image..."
	@docker build -t andros-segmentation:latest .

# Rebuild from scratch (no cache)
rebuild:
	@echo "Rebuilding Docker image from scratch..."
	@docker build --no-cache -t andros-segmentation:latest .

# Run training
train:
	@./docker-run.sh train

# Run evaluation
evaluate:
	@./docker-run.sh evaluate

# Generate masks
generate:
	@./docker-run.sh generate

# Interactive shell
shell:
	@./docker-run.sh shell

# Run tests
test:
	@./docker-run.sh test

# CPU-only training
cpu:
	@./docker-run.sh cpu

# Docker Compose commands
up:
	@docker-compose up

down:
	@docker-compose down

logs:
	@docker-compose logs -f

# Clean up
clean:
	@echo "Cleaning up Docker resources..."
	@docker container prune -f
	@docker image prune -f
	@echo "Done!"
