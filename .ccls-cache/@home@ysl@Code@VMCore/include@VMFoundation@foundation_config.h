

#pragma once

#if defined( _WIN32 ) && defined( VMCORE_SHARED_LIBRARY )
#ifdef vmcore_EXPORTS
#define VMFOUNDATION_EXPORTS __declspec( dllexport )
#else
#define VMFOUNDATION_EXPORTS __declspec( dllimport )
#endif
#else
#define VMFOUNDATION_EXPORTS
#endif
