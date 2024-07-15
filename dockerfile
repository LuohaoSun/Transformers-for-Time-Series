# Use an official Python runtime as the base image
FROM python:lightning

# Set the working directory in the container
WORKDIR /app

# Copy the rest of the application code into the container
COPY . .
RUN pip install momentfm -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install huggingface_hub==0.19.4 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Specify the command to run when the container starts
CMD [ "python3", "scripts/ad_exp_moment.py"]