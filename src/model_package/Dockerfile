# Use a minimal base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire contents of the model_package folder, including data.parquet, into the container
COPY . /app

# Install dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt

# Default command to run the prediction script
ENTRYPOINT ["python", "run_model.py"]