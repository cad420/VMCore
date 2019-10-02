
#include <ostream>
#include <VMFoundation/dataarena.h>
//#include <cmalloc>

//#include <malloc.h>

void *AllocAligned( size_t size, int align )
{
	//#if defined (_WIN32)
	//return _aligned_malloc(size, align);
	//return malloc(size);
	//#else
	return malloc( size );
	//#endif
}

VMFOUNDATION_EXPORTS void FreeAligned( void *ptr )
{
	//#if defined(_WIN32)
	//_aligned_free(ptr);
	//#else
	if ( ptr != nullptr ) {
		free( ptr );
	}
	//#endif
}
