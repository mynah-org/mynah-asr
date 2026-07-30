/* Minimal .npy loader shared by the tests: v1.0, C-order, dtype <f4 or <f8.
 * Always returns double (f4 is upcast). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

double *npy_load_f(const char *path, size_t *n_elems) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) { fclose(f); return NULL; }
    unsigned short hlen;
    if (fread(&hlen, 2, 1, f) != 1) { fclose(f); return NULL; }
    char *hdr = malloc((size_t)hlen + 1);
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); return NULL; }
    hdr[hlen] = '\0';

    int is_f8 = strstr(hdr, "<f8") != NULL;
    if (!is_f8 && !strstr(hdr, "<f4")) {
        fprintf(stderr, "npy: unsupported dtype: %s\n", hdr);
        free(hdr); fclose(f);
        return NULL;
    }
    size_t total = 1;
    const char *sh = strstr(hdr, "'shape': (");
    if (!sh) { free(hdr); fclose(f); return NULL; }
    sh += strlen("'shape': (");
    while (*sh && *sh != ')') {
        if (*sh >= '0' && *sh <= '9') total *= strtoull(sh, (char **)&sh, 10);
        else sh++;
    }
    free(hdr);

    double *out = malloc(total * sizeof(double));
    if (!out) { fclose(f); return NULL; }
    if (is_f8) {
        if (fread(out, 8, total, f) != total) { free(out); fclose(f); return NULL; }
    } else {
        float *tmp = malloc(total * sizeof(float));
        if (!tmp || fread(tmp, 4, total, f) != total) { free(tmp); free(out); fclose(f); return NULL; }
        for (size_t i = 0; i < total; i++) out[i] = (double)tmp[i];
        free(tmp);
    }
    fclose(f);
    *n_elems = total;
    return out;
}
