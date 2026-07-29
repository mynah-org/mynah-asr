#include "weights.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../vendor/cJSON.h"
#include "gguf.h"

struct mynah_asr_safetensors {
    void *map;
    size_t map_len;
    cJSON *header;         /* mantiene vive le stringhe dei nomi */
    mynah_asr_tensor *tensors;
    size_t n_tensors;
    mynah_asr_gguf *gguf;      /* container alternativo: se set, tensors punta lì */
};

static int dtype_of(const char *s, mynah_asr_dtype *out) {
    if (strcmp(s, "F32") == 0) { *out = MYNAH_ASR_DT_F32; return 0; }
    if (strcmp(s, "F64") == 0) { *out = MYNAH_ASR_DT_F64; return 0; }
    if (strcmp(s, "BF16") == 0) { *out = MYNAH_ASR_DT_BF16; return 0; }
    if (strcmp(s, "F16") == 0) { *out = MYNAH_ASR_DT_F16; return 0; }
    if (strcmp(s, "I8") == 0) { *out = MYNAH_ASR_DT_I8; return 0; }
    if (strcmp(s, "U8") == 0) { *out = MYNAH_ASR_DT_U8; return 0; }
    if (strcmp(s, "I64") == 0) { *out = MYNAH_ASR_DT_I64; return 0; } /* BatchNorm num_batches_tracked */
    return -1;
}

mynah_asr_safetensors *mynah_asr_st_open_quiet(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return NULL;
    close(fd);
    return mynah_asr_st_open(path);
}

mynah_asr_safetensors *mynah_asr_st_open(const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "weights: cannot open %s\n", path); return NULL; }
    struct stat sb;
    if (fstat(fd, &sb) != 0 || sb.st_size < 8) { close(fd); return NULL; }

    /* GGUF container? (magic "GGUF"): same API, the consumers do not change */
    char magic[4] = {0};
    if (pread(fd, magic, 4, 0) == 4 && memcmp(magic, "GGUF", 4) == 0) {
        close(fd);
        mynah_asr_gguf *g = mynah_asr_gguf_open(path);
        if (!g) return NULL;
        mynah_asr_safetensors *st = calloc(1, sizeof(*st));
        if (!st) { mynah_asr_gguf_close(g); return NULL; }
        st->gguf = g;
        st->tensors = (mynah_asr_tensor *)mynah_asr_gguf_tensors(g, &st->n_tensors);
        return st;
    }

    void *map = mmap(NULL, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { fprintf(stderr, "weights: mmap failed for %s\n", path); return NULL; }

    uint64_t hlen;
    memcpy(&hlen, map, 8); /* little-endian; assumiamo host LE (arm64/x86_64) */
    if (8 + hlen > (uint64_t)sb.st_size) {
        fprintf(stderr, "weights: header safetensors corrotto in %s\n", path);
        munmap(map, (size_t)sb.st_size);
        return NULL;
    }

    char *hjson = malloc(hlen + 1);
    memcpy(hjson, (const char *)map + 8, hlen);
    hjson[hlen] = '\0';
    cJSON *header = cJSON_Parse(hjson);
    free(hjson);
    if (!header) {
        fprintf(stderr, "weights: header JSON non valido in %s\n", path);
        munmap(map, (size_t)sb.st_size);
        return NULL;
    }

    mynah_asr_safetensors *st = calloc(1, sizeof(*st));
    st->map = map;
    st->map_len = (size_t)sb.st_size;
    st->header = header;

    size_t count = 0;
    for (cJSON *it = header->child; it; it = it->next)
        if (strcmp(it->string, "__metadata__") != 0) count++;
    st->tensors = calloc(count, sizeof(mynah_asr_tensor));

    const uint8_t *base = (const uint8_t *)map + 8 + hlen;
    size_t i = 0;
    for (cJSON *it = header->child; it; it = it->next) {
        if (strcmp(it->string, "__metadata__") == 0) continue;
        mynah_asr_tensor *t = &st->tensors[i];
        t->name = it->string;

        const cJSON *jd = cJSON_GetObjectItemCaseSensitive(it, "dtype");
        const cJSON *js = cJSON_GetObjectItemCaseSensitive(it, "shape");
        const cJSON *jo = cJSON_GetObjectItemCaseSensitive(it, "data_offsets");
        if (!cJSON_IsString(jd) || !cJSON_IsArray(js) || !cJSON_IsArray(jo) ||
            dtype_of(jd->valuestring, &t->dtype) != 0) {
            fprintf(stderr, "weights: voce '%s' non valida (dtype %s)\n", it->string,
                    cJSON_IsString(jd) ? jd->valuestring : "?");
            mynah_asr_st_close(st);
            return NULL;
        }
        t->n_dims = cJSON_GetArraySize(js);
        t->n_elems = 1;
        for (int d = 0; d < t->n_dims && d < 8; d++) {
            t->shape[d] = (int64_t)cJSON_GetArrayItem(js, d)->valuedouble;
            t->n_elems *= (size_t)t->shape[d];
        }
        double off0 = cJSON_GetArrayItem(jo, 0)->valuedouble;
        t->data = base + (size_t)off0;
        i++;
    }
    st->n_tensors = i;
    return st;
}

void mynah_asr_st_close(mynah_asr_safetensors *st) {
    if (!st) return;
    if (st->gguf) {           /* i tensori appartengono al handle GGUF */
        mynah_asr_gguf_close(st->gguf);
        free(st);
        return;
    }
    if (st->map) munmap(st->map, st->map_len);
    cJSON_Delete(st->header);
    free(st->tensors);
    free(st);
}

const mynah_asr_tensor *mynah_asr_st_get(const mynah_asr_safetensors *st, const char *name) {
    for (size_t i = 0; i < st->n_tensors; i++)
        if (strcmp(st->tensors[i].name, name) == 0) return &st->tensors[i];
    return NULL;
}

size_t mynah_asr_st_count(const mynah_asr_safetensors *st) { return st->n_tensors; }

const mynah_asr_tensor *mynah_asr_st_at(const mynah_asr_safetensors *st, size_t i) {
    return i < st->n_tensors ? &st->tensors[i] : NULL;
}
