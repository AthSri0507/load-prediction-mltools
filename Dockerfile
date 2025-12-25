# Use official python image
FROM python:3.10.11-slim

# Set working directory inside container
WORKDIR /app

# Copy project files into container
COPY . .

# Install required packages
RUN pip install --no-cache-dir -r requirements.txt

# Run your training pipeline by default
CMD ["python", "train_nextstep_load3.py", "--csv", "smart_grid_dataset.csv", "--plots", "--mlflow"]
