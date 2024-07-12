# Use an official Python runtime as the base image
FROM python:lightning

# Set the working directory in the container
WORKDIR /app

# Copy the rest of the application code into the container
COPY . .

# Specify the command to run when the container starts
CMD [ "python3", "tutorial/1.快速上手.py"]