/* Utility HTTP/WS del server: SHA-1 + base64 (handshake WebSocket), memmem. */
#ifndef MYNAH_ASR_HTTP_UTIL_H
#define MYNAH_ASR_HTTP_UTIL_H

#include <stddef.h>
#include <stdint.h>

void mynah_asr_sha1(const uint8_t *data, size_t len, uint8_t out[20]);
void mynah_asr_b64(const uint8_t *data, size_t len, char *out /* >= 4*ceil(len/3)+1 */);

/* memmem portabile (aghi piccoli). NULL se assente. */
const uint8_t *mynah_asr_memmem(const uint8_t *hay, size_t hay_len,
                            const uint8_t *needle, size_t needle_len);

#endif
