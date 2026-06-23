#include <stdlib.h>
#include <malloc.h>
#include <errno.h>
#include <unistd.h>

int posix_memalign(void **memptr, size_t alignment, size_t size) {
    void *p = memalign(alignment, size);
    if (!p) {
        return ENOMEM;
    }
    *memptr = p;
    return 0;
}

#define QNX_PAGE_SIZE   4096L
#define QNX_PHYS_BYTES  (4L * 1024 * 1024 * 1024)

long sysconf(int name) {
#ifdef _SC_PAGESIZE
    if (name == _SC_PAGESIZE) return QNX_PAGE_SIZE;
#endif
#if defined(_SC_PAGE_SIZE) && (!defined(_SC_PAGESIZE) || _SC_PAGE_SIZE != _SC_PAGESIZE)
    if (name == _SC_PAGE_SIZE) return QNX_PAGE_SIZE;
#endif
#ifdef _SC_PHYS_PAGES
    if (name == _SC_PHYS_PAGES) return QNX_PHYS_BYTES / QNX_PAGE_SIZE;
#endif
#ifdef _SC_AVPHYS_PAGES
    if (name == _SC_AVPHYS_PAGES) return QNX_PHYS_BYTES / QNX_PAGE_SIZE;
#endif
#ifdef _SC_NPROCESSORS_ONLN
    if (name == _SC_NPROCESSORS_ONLN) return 4;
#endif
#ifdef _SC_NPROCESSORS_CONF
    if (name == _SC_NPROCESSORS_CONF) return 4;
#endif
    errno = EINVAL;
    return -1;
}
