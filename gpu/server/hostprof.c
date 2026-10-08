/* gpu/server/hostprof.c — see hostprof.h. */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE   /* RUSAGE_THREAD, sched_getaffinity, CPU_COUNT */
#endif
#include "hostprof.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>
#if defined(__linux__)
#include <sched.h>
#endif

const char *const HP_NAME[HP__N] = {"scan", "stage_copy", "mel", "idle", "cohort_wait", "step", "publish"};

double hostprof_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void hostprof_lap(hostprof_cycle *c, int ph) {
    const double n = hostprof_now();
    const double us = (n - c->t) * 1e6;
    c->us[ph] += us;
    if (ph != HP_IDLE && ph != HP_COHORT_WAIT && ph != HP_STEP) c->host_us += us;
    c->t = n;
}

static int bucket(double us) {
    int b = (int)(us / HP_HIST_US);
    if (b < 0) b = 0;
    if (b >= HP_HIST_BUCKETS) b = HP_HIST_BUCKETS - 1;
    return b;
}

void hostprof_fold(hostprof *h, hostprof_cycle *c, int stepped, int lanes, double dev_wait_us,
                   double step_start) {
    const double step_us = c->us[HP_STEP];
    for (int i = 0; i < HP__N; i++) { h->us[i] += c->us[i]; c->us[i] = 0.0; }
    h->chunks += c->chunks; c->chunks = 0;
    if (!stepped) return;
    h->cycles++;
    h->lanes += (unsigned long)lanes;
    h->dev_wait_us += dev_wait_us;
    /* host busy in the cycle that ENDS at this step's publish: the server's
     * own host phases since the previous step started, plus the step's wall
     * that was not a device wait */
    double host = c->host_us + step_us - dev_wait_us;
    if (host < 0.0) host = 0.0;
    h->hist_host[bucket(host)]++;
    if (host > h->max_host_us) h->max_host_us = host;
    if (c->cycle_start > 0.0) {
        const double cyc = (step_start - c->cycle_start) * 1e6;
        h->hist_cycle[bucket(cyc)]++;
        if (cyc > h->max_cycle_us) h->max_cycle_us = cyc;
    }
    c->cycle_start = step_start;
    c->host_us = 0.0;
}

void hostprof_rusage(hostprof *h) {
#if defined(__linux__) && defined(RUSAGE_THREAD)
    struct rusage ru;
    if (getrusage(RUSAGE_THREAD, &ru) != 0) return;
    h->utime_s = (double)ru.ru_utime.tv_sec + (double)ru.ru_utime.tv_usec * 1e-6;
    h->stime_s = (double)ru.ru_stime.tv_sec + (double)ru.ru_stime.tv_usec * 1e-6;
    h->nvcsw = ru.ru_nvcsw; h->nivcsw = ru.ru_nivcsw;
    h->minflt = ru.ru_minflt; h->majflt = ru.ru_majflt;
    h->have_rusage = 1;
#else
    (void)h;
#endif
}

double hostprof_quantile(const unsigned long *hist, double q) {
    unsigned long n = 0;
    for (int i = 0; i < HP_HIST_BUCKETS; i++) n += hist[i];
    if (n == 0) return 0.0;
    const double want = q * (double)n;
    unsigned long seen = 0;
    for (int i = 0; i < HP_HIST_BUCKETS; i++) {
        seen += hist[i];
        if ((double)seen >= want) return (double)(i + 1) * HP_HIST_US;   /* bucket upper edge */
    }
    return (double)HP_HIST_BUCKETS * HP_HIST_US;
}

/* a bucket's upper edge, never above the largest value actually seen */
static double qcap(const unsigned long *hist, double q, double max) {
    const double v = hostprof_quantile(hist, q);
    return v < max ? v : max;
}

#define W(...) do { if (k < cap) k += (size_t)snprintf(buf + k, cap - k, __VA_ARGS__); if (k >= cap) k = cap ? cap - 1 : 0; } while (0)

size_t hostprof_report(const hostprof *h, const double *engine_us, const char *const *engine_names,
                       int n_engine, unsigned long passes, unsigned long syncs,
                       unsigned long seq, char *buf, size_t cap) {
    size_t k = 0;
    if (cap) buf[0] = '\0';
    double wall = 0.0;
    for (int i = 0; i < HP__N; i++) wall += h->us[i];
    const double cyc = h->cycles ? (double)h->cycles : 1.0;
    const double lanes = h->lanes ? (double)h->lanes : 1.0;
    const double pw = wall > 0.0 ? 100.0 / wall : 0.0;
    const double waits = h->us[HP_IDLE] + h->us[HP_COHORT_WAIT];
    const double host = wall - waits - h->dev_wait_us;
    W("[HOSTP] seq=%lu summary wall_ms=%.1f cycles=%lu lanes=%lu lanes_per_cycle=%.2f chunks=%lu passes=%lu "
      "syncs=%lu syncs_per_pass=%.1f\n",
      seq, wall * 1e-3, h->cycles, h->lanes, (double)h->lanes / cyc, h->chunks, passes, syncs,
      passes ? (double)syncs / (double)passes : 0.0);
    W("[HOSTP] seq=%lu split host_pct=%.1f device_wait_pct=%.1f cohort_wait_pct=%.1f idle_pct=%.1f "
      "host_us_per_cycle=%.0f device_wait_us_per_cycle=%.0f host_us_per_lane=%.1f\n",
      seq, host * pw, h->dev_wait_us * pw, h->us[HP_COHORT_WAIT] * pw, h->us[HP_IDLE] * pw,
      host / cyc, h->dev_wait_us / cyc, host / lanes);
    for (int i = 0; i < HP__N; i++)
        W("[HOSTP] seq=%lu phase=%-12s kind=%-6s total_ms=%10.1f us_per_cycle=%8.0f us_per_lane=%7.1f pct_wall=%5.1f\n",
          seq, HP_NAME[i], (i == HP_IDLE || i == HP_COHORT_WAIT) ? "wait" : i == HP_STEP ? "step" : "host",
          h->us[i] * 1e-3, h->us[i] / cyc, h->us[i] / lanes, h->us[i] * pw);
    if (engine_us && n_engine > 0) {
        double sum = 0.0;
        for (int i = 0; i < n_engine; i++) sum += engine_us[i];
        for (int i = 0; i < n_engine; i++) {
            const int wait = strstr(engine_names[i], "_wait") != NULL;
            W("[HOSTP] seq=%lu step.%-12s kind=%-6s total_ms=%10.1f us_per_cycle=%8.0f us_per_lane=%7.1f pct_wall=%5.1f\n",
              seq, engine_names[i], wait ? "device" : "host", engine_us[i] * 1e-3, engine_us[i] / cyc,
              engine_us[i] / lanes, engine_us[i] * pw);
        }
        W("[HOSTP] seq=%lu step.sum total_ms=%.1f (step phase %.1f; the difference is the call itself)\n",
          seq, sum * 1e-3, h->us[HP_STEP] * 1e-3);
    }
    W("[HOSTP] seq=%lu tail cycle_us p50=%.0f p95=%.0f p99=%.0f max=%.0f host_us p50=%.0f p95=%.0f p99=%.0f max=%.0f "
      "bucket_us=%.0f\n",
      seq, qcap(h->hist_cycle, 0.5, h->max_cycle_us), qcap(h->hist_cycle, 0.95, h->max_cycle_us),
      qcap(h->hist_cycle, 0.99, h->max_cycle_us), h->max_cycle_us, qcap(h->hist_host, 0.5, h->max_host_us),
      qcap(h->hist_host, 0.95, h->max_host_us), qcap(h->hist_host, 0.99, h->max_host_us), h->max_host_us, HP_HIST_US);
    if (h->have_rusage)
        W("[HOSTP] seq=%lu rusage engine_thread utime_s=%.2f stime_s=%.2f cpu_pct_of_wall=%.1f nvcsw=%ld nivcsw=%ld "
          "minflt=%ld majflt=%ld\n",
          seq, h->utime_s, h->stime_s, wall > 0.0 ? (h->utime_s + h->stime_s) * 1e8 / wall : 0.0,
          h->nvcsw, h->nivcsw, h->minflt, h->majflt);
    return k;
}

size_t hostprof_metrics(const hostprof *h, const double *engine_us, const char *const *engine_names,
                        int n_engine, char *buf, size_t cap) {
    size_t k = 0;
    if (cap) buf[0] = '\0';
    W("# TYPE mynah_asr_gpu_hostp_us_total counter\n");
    for (int i = 0; i < HP__N; i++) W("mynah_asr_gpu_hostp_us_total{phase=\"%s\"} %.0f\n", HP_NAME[i], h->us[i]);
    for (int i = 0; engine_us && i < n_engine; i++)
        W("mynah_asr_gpu_hostp_us_total{phase=\"step.%s\"} %.0f\n", engine_names[i], engine_us[i]);
    W("# TYPE mynah_asr_gpu_hostp_cycles_total counter\nmynah_asr_gpu_hostp_cycles_total %lu\n", h->cycles);
    W("# TYPE mynah_asr_gpu_hostp_lanes_total counter\nmynah_asr_gpu_hostp_lanes_total %lu\n", h->lanes);
    W("# TYPE mynah_asr_gpu_hostp_device_wait_us_total counter\nmynah_asr_gpu_hostp_device_wait_us_total %.0f\n",
      h->dev_wait_us);
    return k;
}

/* ------------------------------------------------------------- topology */

static int read_first_line(const char *path, char *out, size_t cap) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    if (!fgets(out, (int)cap, f)) { fclose(f); return -1; }
    fclose(f);
    out[strcspn(out, "\n")] = '\0';
    return 0;
}

static void cpu_model(char *out, size_t cap) {
    snprintf(out, cap, "?");
#if defined(__linux__)
    FILE *f = fopen("/proc/cpuinfo", "r");
    if (!f) return;
    char line[512];
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "model name", 10) == 0) {
            const char *c = strchr(line, ':');
            if (c) {
                c++;
                while (*c == ' ' || *c == '\t') c++;
                snprintf(out, cap, "%s", c);
                out[strcspn(out, "\n")] = '\0';
            }
            break;
        }
    }
    fclose(f);
#endif
}

size_t host_topology_line(const char *gpu_pci_bus_id, char *buf, size_t cap) {
    size_t k = 0;
    if (cap) buf[0] = '\0';
    char model[160];
    cpu_model(model, sizeof(model));
    for (char *p = model; *p; p++) if (*p == '"') *p = '\'';
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    int usable = (int)online;
    char mask[256] = "?";
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        usable = CPU_COUNT(&set);
        /* the mask as ranges, e.g. 0-31,64-95 */
        size_t m = 0;
        mask[0] = '\0';
        for (int c = 0; c < CPU_SETSIZE && m + 16 < sizeof(mask); c++) {
            if (!CPU_ISSET(c, &set)) continue;
            int e = c;
            while (e + 1 < CPU_SETSIZE && CPU_ISSET(e + 1, &set)) e++;
            m += (size_t)snprintf(mask + m, sizeof(mask) - m, "%s%d", m ? "," : "", c);
            if (e > c) m += (size_t)snprintf(mask + m, sizeof(mask) - m, "-%d", e);
            c = e;
        }
    }
#endif
    /* the cgroup v2 quota ("max 100000" = none), else v1 */
    char quota[64] = "none";
    char q[128];
    if (read_first_line("/sys/fs/cgroup/cpu.max", q, sizeof(q)) == 0) {
        char a[32] = "", b[32] = "";
        if (sscanf(q, "%31s %31s", a, b) == 2 && strcmp(a, "max") != 0 && atof(b) > 0.0)
            snprintf(quota, sizeof(quota), "%.2f", atof(a) / atof(b));
    } else {
        char a[32], b[32];
        if (read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", a, sizeof(a)) == 0 &&
            read_first_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us", b, sizeof(b)) == 0 && atol(a) > 0 && atol(b) > 0)
            snprintf(quota, sizeof(quota), "%.2f", (double)atol(a) / (double)atol(b));
    }
    char nodes[64] = "?";
    (void)read_first_line("/sys/devices/system/node/online", nodes, sizeof(nodes));
    char gnode[16] = "?";
    if (gpu_pci_bus_id && gpu_pci_bus_id[0]) {
        char id[64], path[160];
        snprintf(id, sizeof(id), "%s", gpu_pci_bus_id);
        for (char *p = id; *p; p++) *p = (char)tolower((unsigned char)*p);
        /* the runtime prints an 8-digit domain (00000000:01:00.0); sysfs uses 4 */
        const char *s = id;
        if (strlen(id) > 12 && strncmp(id, "0000", 4) == 0 && id[8] == ':') s = id + 4;
        snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/numa_node", s);
        (void)read_first_line(path, gnode, sizeof(gnode));
    }
    struct rlimit rl;
    char nofile[48] = "?";
    if (getrlimit(RLIMIT_NOFILE, &rl) == 0) {
        if (rl.rlim_cur == RLIM_INFINITY) snprintf(nofile, sizeof(nofile), "unlimited");
        else snprintf(nofile, sizeof(nofile), "%llu", (unsigned long long)rl.rlim_cur);
    }
    W("[TOPOLOGY] cpu_model=\"%s\" online_cpus=%ld usable_cpus=%d affinity=%s cgroup_quota_cpus=%s numa_nodes=%s "
      "gpu_pci=%s gpu_numa_node=%s nofile_soft=%s\n",
      model, online, usable, mask, quota, nodes, gpu_pci_bus_id && gpu_pci_bus_id[0] ? gpu_pci_bus_id : "-", gnode,
      nofile);
    return k;
}
#undef W
