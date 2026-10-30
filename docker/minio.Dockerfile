FROM golang:1.24-bookworm AS builder

ARG MINIO_VERSION=RELEASE.2025-09-07T16-13-09Z

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Build the exact official release tag and reproduce MinIO's release linker
# metadata instead of using plain `go install`, which reports DEVELOPMENT.GOGET.
RUN git clone --depth 1 --branch "${MINIO_VERSION}" https://github.com/minio/minio.git . \
    && mkdir -p /out \
    && LDFLAGS="$(MINIO_RELEASE=RELEASE go run buildscripts/gen-ldflags.go)" \
    && CGO_ENABLED=0 go build -tags kqueue -trimpath --ldflags "${LDFLAGS}" -o /out/minio . \
    && /out/minio --version | grep -F "minio version ${MINIO_VERSION}"

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data \
    && chown -R app:app /data

COPY --from=builder --chown=app:app /out/minio /usr/local/bin/minio

USER app

EXPOSE 9000 9001

ENTRYPOINT ["/usr/local/bin/minio"]
CMD ["server", "/data", "--console-address", ":9001"]
