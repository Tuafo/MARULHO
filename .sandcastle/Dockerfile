FROM node:22-bookworm

# Install system dependencies
RUN apt-get update && apt-get install -y \
git \
curl \
jq \
python3 \
python3-pip \
python3-venv \
&& rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
| dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
| tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& apt-get update && apt-get install -y gh \
&& rm -rf /var/lib/apt/lists/*

# Rename the base image's "node" user (UID 1000) to "agent".
RUN usermod -d /home/agent -m -l agent node

# Install Pi coding agent (run as root before USER agent)
RUN npm install -g @mariozechner/pi-coding-agent

RUN python3 -m venv /opt/venv && chown -R agent:node /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Pre-install Python deps as "agent" user so venv stays writable at runtime
COPY --chown=agent:node pyproject.toml /tmp/preinstall/pyproject.toml
RUN mkdir -p /tmp/preinstall/src/hecsn && touch /tmp/preinstall/src/hecsn/__init__.py \
&& chown -R agent:node /tmp/preinstall

USER agent

# Configure Pi to use NVIDIA NIM as a custom OpenAI-compatible provider
COPY --chown=agent:node .sandcastle/pi-models.json /home/agent/.pi/agent/models.json

RUN /opt/venv/bin/pip install --no-cache-dir -e '/tmp/preinstall[dev]' \
    && rm -rf /tmp/preinstall

WORKDIR /home/agent

# In worktree sandbox mode, Sandcastle bind-mounts the git worktree at /home/agent/workspace
# and overrides the working directory to /home/agent/workspace at container start.
# Structure your Dockerfile so that /home/agent/workspace can serve as the project root.
ENTRYPOINT ["sleep", "infinity"]
