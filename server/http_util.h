/* Server HTTP/WS utilities: SHA-1 + base64 (WebSocket handshake), memmem. */
#ifndef MYNAH_ASR_HTTP_UTIL_H
#define MYNAH_ASR_HTTP_UTIL_H

#include <stddef.h>
#include <stdint.h>

void mynah_asr_sha1(const uint8_t *data, size_t len, uint8_t out[20]);
void mynah_asr_b64(const uint8_t *data, size_t len, char *out /* >= 4*ceil(len/3)+1 */);

/* Portable memmem (small needles). NULL when not found. */
const uint8_t *mynah_asr_memmem(const uint8_t *hay, size_t hay_len,
                            const uint8_t *needle, size_t needle_len);


/* Names the CALLING thread for the OS so `top -H` and /proc/<pid>/task show
 * which thread is hot without a debugger. Diagnostic: failure is ignored.
 * Linux caps the name at 15 bytes plus the terminator; longer names are
 * truncated here rather than rejected there. */
void mynah_asr_thread_set_name(const char *name);

#endif
