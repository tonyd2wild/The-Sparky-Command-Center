# Sparky Command Center - container image.
#
# The dashboard itself has no third-party Python dependencies, but it shells out
# to `ssh` for every node poll, so the image needs an SSH client.
#
# It also needs a real /etc/passwd entry for the uid it runs as: OpenSSH refuses
# to start for a uid with no user record ("No user exists for uid 1000"), which
# would make every node poll fail while the model cards kept working. Hence the
# `sparky` account below, created with uid/gid 1000 so a bind-mounted key from a
# typical single-user host is readable as-is.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 1000 sparky \
    && useradd -u 1000 -g 1000 -m -d /home/sparky -s /bin/bash sparky

WORKDIR /app
COPY server.py config.example.json ./
RUN mkdir -p /app/data && chown -R sparky:sparky /app/data

# HOME matters: `~/.ssh/...` paths in config.json expand against it, so the same
# config file works both on the host and in the container.
ENV HOME=/home/sparky \
    CONFIG=/app/config.json \
    PYTHONUNBUFFERED=1

USER sparky
EXPOSE 8890
CMD ["python3", "server.py"]
