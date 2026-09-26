/* gpu/server/ws.h — the WebSocket server side the GPU server needs: the RFC
 * 6455 handshake and the reading of masked client frames. Server-to-client
 * frames are built by mynah_asr_ws_frame (server/http_util.h, reused).
 *
 * Kept deliberately small and its own: the CPU server's framing lives inside
 * server/main.c as static functions and is not modified for this tree. The
 * behaviour is the same, tested by the same probes (tests/ws_probe.py,
 * tests/fault_probe.py). */
#ifndef MYNAH_ASR_GPU_WS_H
#define MYNAH_ASR_GPU_WS_H

#include <stddef.h>
#include <stdint.h>

enum { WS_READ_OK = 0, WS_READ_EOF, WS_READ_TIMEOUT, WS_READ_ERROR };

/* Read exactly n bytes (EINTR-safe). SO_RCVTIMEO on the fd turns an idle
 * client into WS_READ_TIMEOUT. */
int ws_read_exact(int fd, uint8_t *buf, size_t n);

/* The Sec-WebSocket-Accept value for a client key, into out (>= 32 bytes). */
void ws_accept_key(const char *client_key, char *out, size_t cap);

typedef struct {
    int opcode, fin, rsv;        /* rsv != 0 -> protocol error */
    int masked;
    uint64_t len;
    uint8_t mask[4];
} ws_frame_hdr;

/* Reads one frame header. Returns a WS_READ_* code. */
int ws_read_header(int fd, ws_frame_hdr *h);

#endif
