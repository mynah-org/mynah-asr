#!/usr/bin/env python3
"""configs/observability/ against the server source (stdlib only, no server).

Fails when a series that a dashboard panel, an alert or the OTel collector
config names is not one the C source renders: a rename in C would otherwise
empty a panel or disarm an alert in silence. Also checks that every `le`
threshold an alert counts on is a real histogram edge (an `le` that is not an
edge matches no series and the alert never fires), that the dashboard is valid
JSON, and, when PyYAML is importable, that the YAML files parse.

Exit 0 ok, 1 otherwise.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OBS = ROOT / "configs" / "observability"
SOURCES = sorted((ROOT / "server").glob("*.c")) + [ROOT / "gpu" / "server" / "main.c"]
NAME = re.compile(r"mynah_asr_[a-z0-9_]+")
HIST_SUFFIX = ("_bucket", "_count", "_sum")


def main() -> int:
    failed = 0
    src = "\n".join(p.read_text(encoding="utf-8") for p in SOURCES)
    rendered = set(NAME.findall(src))

    files = [OBS / "alerts.yml", OBS / "prometheus.yml", OBS / "otel-collector.yaml",
             OBS / "grafana" / "dashboards" / "mynah-asr.json"]
    try:
        json.loads(files[-1].read_text(encoding="utf-8"))
    except ValueError as e:
        print(f"FAIL dashboard is not valid JSON: {e}")
        failed += 1
    try:
        import yaml  # noqa: PLC0415 -- optional
        for f in files[:3] + sorted((OBS / "grafana" / "provisioning").rglob("*.yml")) + \
                [OBS / "docker-compose.yml"]:
            yaml.safe_load(f.read_text(encoding="utf-8"))
    except ImportError:
        print("note: PyYAML not importable, YAML syntax not checked")
    except Exception as e:  # a parse error names the file in its message
        print(f"FAIL YAML: {e}")
        failed += 1

    named = 0
    for f in files:
        for n in sorted(set(NAME.findall(f.read_text(encoding="utf-8")))):
            named += 1
            base = n
            for suf in HIST_SUFFIX:
                if n.endswith(suf) and n not in rendered:
                    base = n[: -len(suf)]
            if base not in rendered:
                print(f"FAIL {f.relative_to(ROOT)}: {n} is not rendered by the server source")
                failed += 1

    # the le thresholds the alerts count on must be histogram edges
    edges_ms = re.search(r"mynah_asr_fleet_ms_edges\[[^]]*\]\s*=\s*\{([^}]*)\}",
                         (ROOT / "server" / "fleet.c").read_text(encoding="utf-8"))
    edges = {float(x) / 1000.0 for x in edges_ms.group(1).split(",")} if edges_ms else set()
    for le in re.findall(r'le="([0-9.]+)"', (OBS / "alerts.yml").read_text(encoding="utf-8")):
        if float(le) not in edges:
            print(f"FAIL alerts.yml: le=\"{le}\" is not an edge of the ms histograms {sorted(edges)}")
            failed += 1

    print(f"observability configs: {named} series names checked, {failed} failure(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
