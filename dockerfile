# DESCRIPTION:    
# This Dockerfile is used to build an image containing lightning.

FROM python:lightning-main

WORKDIR /

# RUN curl -fsSL https://code-server.dev/install.sh | sh -s -- --dry-run

EXPOSE 22

CMD ["python"]
# CMD ["code-server", "--auth", "none", "--port", "8080", "--host", "0.0.0.0"]