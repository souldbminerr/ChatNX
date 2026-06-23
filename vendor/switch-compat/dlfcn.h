#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#ifndef RTLD_LAZY
#define RTLD_LAZY   0x0001
#endif
#ifndef RTLD_NOW
#define RTLD_NOW    0x0002
#endif
#ifndef RTLD_LOCAL
#define RTLD_LOCAL  0x0000
#endif
#ifndef RTLD_GLOBAL
#define RTLD_GLOBAL 0x0100
#endif

static inline void *dlopen(const char *filename, int flag) {
    (void)filename;
    (void)flag;
    return 0;
}

static inline int dlclose(void *handle) {
    (void)handle;
    return 0;
}

static inline void *dlsym(void *handle, const char *symbol) {
    (void)handle;
    (void)symbol;
    return 0;
}

static inline const char *dlerror(void) {
    return "not supported";
}

#ifdef __cplusplus
}
#endif
