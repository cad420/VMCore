#pragma once

#if defined(__linux) || defined(__linux__) || defined(linux)
# define VM_OS_STR "LINUX"
# define VM_OS_LINUX
#elif defined(__APPLE__)
# define VM_OS_STR "MACOS"
# define VM_OS_MACOS 
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64)
# define VM_OS_STR "WINDOWS"
# define VM_OS_WIN
#endif
