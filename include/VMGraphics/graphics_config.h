
#ifndef _GRAPHICS_CONFIG_H_
#define _GRAPHICS_CONFIG_H_

#if defined( _WIN32 ) && defined(VMCORE_SHARED_LIBRARY)
#ifdef vmgraphics_EXPORTS
#define VMGRAPHICS_EXPORTS __declspec( dllexport )
#else
#define VMGRAPHICS_EXPORTS __declspec( dllimport )
#endif
#else
#define VMGRAPHICS_EXPORTS
#endif

#endif