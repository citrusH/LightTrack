#ifndef COMMON_H
#define COMMON_H

#define RETURN_1(format, ...) do { \
    printf("[aitrack %s %d] " format "\n", __func__, __LINE__, ##__VA_ARGS__); \
    return 1; \
} while(0)

#define RETURN_NULL(format, ...) do { \
    printf("[aitrack %s %d] " format "\n", __func__, __LINE__, ##__VA_ARGS__); \
    return NULL; \
} while(0)



#endif 