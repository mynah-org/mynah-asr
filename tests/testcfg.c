/* Shared helper for the parity tests: reads from mynah.json the parameters that
 * vary across models (normalize, default att context, it-IT prompt).
 * Models without streaming/prompt (Parakeet): left/right/prompt = -1. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../vendor/cJSON.h"

int test_model_cfg(const char *model_dir, int *normalize_pf, int *left, int *right,
                   int *prompt_it) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/mynah.json", model_dir);
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)len + 1);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { free(buf); fclose(f); return -1; }
    buf[len] = '\0';
    fclose(f);
    cJSON *cfg = cJSON_Parse(buf);
    free(buf);
    if (!cfg) return -1;

    const cJSON *jn = cJSON_GetObjectItem(cJSON_GetObjectItem(cfg, "features"), "normalize");
    *normalize_pf = jn && cJSON_IsString(jn) && strcmp(jn->valuestring, "per_feature") == 0;

    *left = -1; *right = -1; *prompt_it = -1;
    const cJSON *js = cJSON_GetObjectItem(cfg, "streaming");
    if (js) {
        const int def = cJSON_GetObjectItem(js, "default_preset_index")->valueint;
        const cJSON *p = cJSON_GetArrayItem(cJSON_GetObjectItem(js, "att_context_presets"), def);
        *left = cJSON_GetArrayItem(p, 0)->valueint;
        *right = cJSON_GetArrayItem(p, 1)->valueint;
    }
    const cJSON *jp = cJSON_GetObjectItem(cfg, "prompt");
    if (jp) {
        const cJSON *e = cJSON_GetObjectItem(cJSON_GetObjectItem(jp, "dictionary"), "it-IT");
        if (e) *prompt_it = e->valueint;
    }
    cJSON_Delete(cfg);
    return 0;
}
