# Arm motion planning through a lidar OctoMap: FCL + OMPL Informed RRT*, and MoveIt 2.
# Same image locally and on AWS:
#   docker build -t arm-motion-planning .
#   docker run --rm -v "$PWD/results:/work/results" arm-motion-planning
FROM ubuntu:24.04

ARG MAMBA_VERSION=2.10.0
ENV DEBIAN_FRONTEND=noninteractive \
    MAMBA_ROOT_PREFIX=/opt/conda \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    MPLBACKEND=Agg

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl bzip2 \
 && rm -rf /var/lib/apt/lists/*

# micromamba from its conda-forge package (no installer script needed)
RUN curl -fsSL "https://conda.anaconda.org/conda-forge/linux-64/micromamba-${MAMBA_VERSION}-0.tar.bz2" \
    | tar -xj -C /usr/local bin/micromamba

COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -f /tmp/environment.yml \
 && micromamba clean -a -y

WORKDIR /work
COPY sim/ /work/sim/
COPY entrypoint.sh /work/entrypoint.sh
RUN chmod +x /work/entrypoint.sh

# Every run writes to /work/results (mount a host folder there).
ENTRYPOINT ["/work/entrypoint.sh"]
CMD ["all"]
