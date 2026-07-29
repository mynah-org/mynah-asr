#include "http_util.h"

#include <string.h>

/* SHA-1 (RFC 3174) — compatta, solo per l'handshake WebSocket. */
void mynah_asr_sha1(const uint8_t *data, size_t len, uint8_t out[20]) {
    uint32_t h[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
    const uint64_t total_bits = (uint64_t)len * 8;

    uint8_t block[64];
    size_t i = 0;
    int done_pad = 0, done_len = 0;
    while (!done_len) {
        size_t n = 0;
        if (i < len) {
            n = len - i < 64 ? len - i : 64;
            memcpy(block, data + i, n);
            i += n;
        }
        if (n < 64) {
            if (!done_pad) { block[n++] = 0x80; done_pad = 1; }
            if (n <= 56) {
                memset(block + n, 0, 56 - n);
                for (int b = 0; b < 8; b++) block[56 + b] = (uint8_t)(total_bits >> (56 - 8 * b));
                done_len = 1;
            } else {
                memset(block + n, 0, 64 - n);
            }
        }
        uint32_t w[80];
        for (int t = 0; t < 16; t++)
            w[t] = ((uint32_t)block[t * 4] << 24) | ((uint32_t)block[t * 4 + 1] << 16) |
                   ((uint32_t)block[t * 4 + 2] << 8) | block[t * 4 + 3];
        for (int t = 16; t < 80; t++) {
            uint32_t v = w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16];
            w[t] = (v << 1) | (v >> 31);
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int t = 0; t < 80; t++) {
            uint32_t f, k;
            if (t < 20) { f = (b & c) | ((~b) & d); k = 0x5A827999; }
            else if (t < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
            else if (t < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
            else { f = b ^ c ^ d; k = 0xCA62C1D6; }
            uint32_t tmp = ((a << 5) | (a >> 27)) + f + e + k + w[t];
            e = d; d = c; c = (b << 30) | (b >> 2); b = a; a = tmp;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }
    for (int j = 0; j < 5; j++) {
        out[j * 4] = (uint8_t)(h[j] >> 24);
        out[j * 4 + 1] = (uint8_t)(h[j] >> 16);
        out[j * 4 + 2] = (uint8_t)(h[j] >> 8);
        out[j * 4 + 3] = (uint8_t)h[j];
    }
}

void mynah_asr_b64(const uint8_t *data, size_t len, char *out) {
    static const char tab[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t o = 0;
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)data[i] << 16;
        if (i + 1 < len) v |= (uint32_t)data[i + 1] << 8;
        if (i + 2 < len) v |= data[i + 2];
        out[o++] = tab[(v >> 18) & 63];
        out[o++] = tab[(v >> 12) & 63];
        out[o++] = i + 1 < len ? tab[(v >> 6) & 63] : '=';
        out[o++] = i + 2 < len ? tab[v & 63] : '=';
    }
    out[o] = '\0';
}

const uint8_t *mynah_asr_memmem(const uint8_t *hay, size_t hay_len,
                            const uint8_t *needle, size_t needle_len) {
    if (needle_len == 0 || hay_len < needle_len) return NULL;
    for (size_t i = 0; i + needle_len <= hay_len; i++)
        if (hay[i] == needle[0] && memcmp(hay + i, needle, needle_len) == 0)
            return hay + i;
    return NULL;
}
