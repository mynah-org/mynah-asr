/* gpu/server/http_io.h — the response writers of gpu/server/main.c, shared with
 * the offline REST module (rest.c) so both answer with the same framing, the
 * same refusal discipline (write, half-close, drain: a refusal the client can
 * READ) and the same refusal books. */
#ifndef MYNAH_ASR_GPU_SERVER_HTTP_IO_H
#define MYNAH_ASR_GPU_SERVER_HTTP_IO_H

#include <stddef.h>

#include "cJSON.h"

int write_all(int fd, const void *buf, size_t n);
/* write, half-close, drain briefly, close */
void linger_close(int fd, const char *resp, size_t n);
/* an OpenAI-style error body; retry_after > 0 adds Retry-After and counts the
 * refusal as server_at_capacity. Closes fd. */
void refuse_json(int fd, int code, const char *status, const char *type,
                 const char *errcode, const char *msg, int retry_after);
/* 200/other with a JSON body; does NOT close fd */
void send_json(int fd, int code, cJSON *j);
void send_text(int fd, int code, const char *ctype, const char *body, size_t len);

#endif
