/* gpu/server/ws.c — see ws.h. */
#include "ws.h"

#include "http_util.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>

int ws_read_exact(int fd, uint8_t *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        const ssize_t r = recv(fd, buf + got, n - got, 0);
        if (r < 0 && errno == EINTR) continue;
        if (r == 0) return WS_READ_EOF;
        if (r < 0) return (errno == EAGAIN || errno == EWOULDBLOCK) ? WS_READ_TIMEOUT : WS_READ_ERROR;
        got += (size_t)r;
    }
    return WS_READ_OK;
}

void ws_accept_key(const char *client_key, char *out, size_t cap) {
    char src[160];
    snprintf(src, sizeof(src), "%s258EAFA5-E914-47DA-95CA-C5AB0DC85B11", client_key);
    uint8_t sha[20];
    mynah_asr_sha1((const uint8_t *)src, strlen(src), sha);
    char b64[40];
    mynah_asr_b64(sha, 20, b64);
    snprintf(out, cap, "%s", b64);
}

int ws_read_header(int fd, ws_frame_hdr *h) {
    uint8_t b[2];
    int rr = ws_read_exact(fd, b, 2);
    if (rr != WS_READ_OK) return rr;
    h->fin = (b[0] & 0x80) != 0;
    h->rsv = b[0] & 0x70;
    h->opcode = b[0] & 0x0F;
    h->masked = (b[1] & 0x80) != 0;
    h->len = b[1] & 0x7F;
    if (h->len == 126) {
        uint8_t e[2];
        if ((rr = ws_read_exact(fd, e, 2)) != WS_READ_OK) return rr;
        h->len = ((uint64_t)e[0] << 8) | e[1];
    } else if (h->len == 127) {
        uint8_t e[8];
        if ((rr = ws_read_exact(fd, e, 8)) != WS_READ_OK) return rr;
        h->len = 0;
        for (int i = 0; i < 8; i++) h->len = (h->len << 8) | e[i];
    }
    memset(h->mask, 0, 4);
    if (h->masked && (rr = ws_read_exact(fd, h->mask, 4)) != WS_READ_OK) return rr;
    return WS_READ_OK;
}
