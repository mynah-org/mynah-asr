# Observability: Prometheus, Grafana, alerts, OpenTelemetry

The server exposes one Prometheus text page; everything here reads that page.
Nothing in the server pushes, links an SDK or runs a collector thread.

```sh
# a server with its metrics on loopback (the GPU server takes the same flag)
./mynah-asr-server -m models/nemotron-3.5-asr-streaming-0.6b --prefork 3 --metrics-port 9101
./mynah-asr-server-cuda -m models/nemotron-3.5-asr-streaming-0.6b --metrics-port 9102

configs/observability/observability_up.sh          # Prometheus :9090 + Grafana :3000
configs/observability/observability_up.sh up --otel # + an OTel collector forwarding over OTLP
configs/observability/observability_up.sh down
```

| file | what it is |
|---|---|
| `prometheus.yml` | one scrape target per server process. Under `--prefork` the router answers and carries the summed `mynah_asr_fleet_*` series, so a fleet is still one target |
| `alerts.yml` | integrity (books, leaked slots, dead workers, lost sessions, CUDA errors) and service (late deltas, slow finals, backlog, 503s, saturation) |
| `grafana/dashboards/mynah-asr.json` | one row per operational question (docs/server.md, "A first view") |
| `grafana/provisioning/` | the datasource and the dashboard, provisioned at start |
| `otel-collector.yaml` | the same page through the collector's `prometheus` receiver, exported with `otlphttp` to `$OTEL_EXPORTER_OTLP_ENDPOINT` |
| `docker-compose.yml`, `observability_up.sh` | all of it on loopback, host networking (Linux) |

**The rules the configs follow** (they are the server's, `docs/server.md`):

- An alert fires on something a listener would notice, and its threshold is
  an exact count. The emission-lag, finalization and first-text histograms
  have edges that are multiples of the scheduler's 8 ms bucket, so "deltas
  320 ms late or worse" is counted, not interpolated. The quantile panels are
  for the eye only.
- Client-side latency (TTFP as a client measures it, stalls) is not on the
  page and not on the dashboard: it belongs to the load harness
  (`tools/bench/`), which owns the far end of the socket.
- Cardinality is fixed: the labels are `worker`, `reason`, `code`, `le`, plus
  `model` in a multi-model fleet (the operator's `--model` names). Never a
  language, a path or anything a client chooses.
- `/metrics` is bound to 127.0.0.1 and refuses more than 5 scrapes/s (burst
  10): scrape every 5 s from the same host, or put an agent there (the
  collector above is one).

The GPU server (`mynah-asr-server-cuda`) is one process and renders its own
smaller set (`mynah_asr_sessions_*{worker="gpu"}`, `mynah_asr_gpu_*`); the
dashboard plots both servers where the meaning is the same.

`tests/test_observability_configs.py` (in `make check`) fails when a series
named here is not one the server source renders, so a rename in C cannot
silently empty a panel or disarm an alert.
