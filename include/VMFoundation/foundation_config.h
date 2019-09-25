
#ifndef _CORE_CONFIG_H_
#define _CORE_CONFIG_H_

#if defined( _WIN32 ) && defined( VMCORE_SHARED_LIBRARY )
#ifdef vmfoundation_EXPORTS
#define VMFOUNDATION_EXPORTS __declspec( dllexport )
#else
#define VMFOUNDATION_EXPORTS __declspec( dllimport )
#endif
#else
#define VMFOUNDATION_EXPORTS
#endif

#endif