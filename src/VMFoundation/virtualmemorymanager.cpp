
#include <cstring>
#include <VMFoundation/virtualmemorymanager.h>

namespace ysl
{
void AbstrMemoryCache::SetNextLevelCache( IPageFile *cache )
{
	nextLevel = cache;
}

void AbstrMemoryCache::SetCachePolicy( AbstrCachePolicy *policy )
{
	if ( !policy ) return;
	if ( cachePolicy ) {
		cachePolicy->SetOwnerCache( nullptr );
	}
	cachePolicy = policy;
	cachePolicy->SetOwnerCache( this );
	cachePolicy->InitEvent( this );
}

AbstrCachePolicy *AbstrMemoryCache::TakeCachePolicy()
{
	if ( cachePolicy ) {
		cachePolicy->SetOwnerCache( nullptr );
		return cachePolicy;
	}
	return nullptr;
}

const void *AbstrMemoryCache::GetPage( size_t pageID )
{
	assert( cachePolicy );
	const bool e = cachePolicy->QueryPage( pageID );
	if ( !e ) {
		const auto storageID = cachePolicy->QueryAndUpdate( pageID );
		// Read block from next level to the storage cache
		const auto storage = GetPageStorage_Implement( storageID );

		memcpy( storage, nextLevel->GetPage( pageID ), GetPageSize() );
		return storage;
	} else {
		const auto storageID = cachePolicy->QueryAndUpdate( pageID );
		return GetPageStorage_Implement( storageID );
	}
}

}  // namespace ysl
