
#include <cstring>
#include <VMFoundation/virtualmemorymanager.h>
#include <VMUtils/ref.hpp>
#include <VMFoundation/logger.h>
#include <unordered_set>

namespace vm
{
class AbstrMemoryCache__pImpl
{
	//AbstrMemoryCache * const q_ptr = nullptr;
	VM_DECL_API( AbstrMemoryCache )
public:
	AbstrMemoryCache__pImpl( AbstrMemoryCache *api ) :
	  q_ptr( api ) {}
	Ref<IPageFile> nextLevel;
	Ref<AbstrCachePolicy> cachePolicy;
	std::unordered_set<size_t> dirtyPageID;
};


AbstrMemoryCache::AbstrMemoryCache( IRefCnt *cnt ) :
  ::vm::EverythingBase<IPageFile>( cnt ), d_ptr( new AbstrMemoryCache__pImpl( this ) )
{
}

void AbstrMemoryCache::SetNextLevelCache( IPageFile *cache )
{
	VM_IMPL( AbstrMemoryCache );

	_->nextLevel = cache;
}

void AbstrMemoryCache::SetCachePolicy( AbstrCachePolicy *policy )
{
	VM_IMPL( AbstrMemoryCache );

	if ( !policy ) return;
	if ( _->cachePolicy ) {
		_->cachePolicy->SetOwnerCache( nullptr );
	}
	_->cachePolicy = policy;
	_->cachePolicy->SetOwnerCache( this );
	_->cachePolicy->InitEvent( this );
}

AbstrCachePolicy *AbstrMemoryCache::TakeCachePolicy()
{
	VM_IMPL( AbstrMemoryCache );
	if ( _->cachePolicy ) {
		_->cachePolicy->SetOwnerCache( nullptr );
		return _->cachePolicy;
	}
	return nullptr;
}

IPageFile *AbstrMemoryCache::GetNextLevelCache()
{
	VM_IMPL( AbstrMemoryCache )
	return _->nextLevel;
}

const IPageFile *AbstrMemoryCache::GetNextLevelCache() const
{
	const auto _ = d_func();
	return _->nextLevel;
}

const void *AbstrMemoryCache::GetPage( size_t pageID )
{
	VM_IMPL( AbstrMemoryCache )
	assert( _->cachePolicy );
	const bool e = _->cachePolicy->QueryPage( pageID );
	if ( !e ) {
		//const auto storageID = _->cachePolicy->QueryAndUpdate( pageID );
		size_t storageID, evictedPageID;
		_->cachePolicy->QueryAndUpdate( pageID, &storageID, &evictedPageID );

		// Read block from next level to the storage cache
		const auto storage = GetPageStorage_Implement( storageID );

		memcpy( storage, _->nextLevel->GetPage( pageID ), GetPageSize() );
		return storage;
	} else {
		size_t storageID, evictedPageID;
		_->cachePolicy->QueryAndUpdate( pageID, &storageID, &evictedPageID );
		//const auto storageID = _->cachePolicy->QueryAndUpdate( pageID );
		return GetPageStorage_Implement( storageID );
	}
}

void AbstrMemoryCache::Flush()
{
	//TODO: Flush all dirty page and reset dirty
	VM_IMPL( AbstrMemoryCache )
}

void AbstrMemoryCache::Write( const void *page, size_t pageID, bool flush )
{
	VM_IMPL( AbstrMemoryCache )
	// Only suport write through now
	if ( flush ) {
		//read, update and write
		auto cachedPage = const_cast<void *>( GetPage( pageID ) );
		memcpy( cachedPage, page, GetPageSize() );
		_->nextLevel->Write( page, pageID, flush );	 // update next level cache
	} else {
		LOG_WARNING << "Only support write through only";
		//TODO: set dirty flag
		_->dirtyPageID.insert(pageID);
	}
}

void AbstrMemoryCache::Flush( size_t pageID )
{
	//TODO :: write to the next level cache and flush
	LOG_WARNING << "Not implemented yet";
}

AbstrMemoryCache::~AbstrMemoryCache()
{
}

void AbstrMemoryCache::Replace_Event( size_t evictPageID )
{

}

class AbstrCachePolicy__pImpl
{
	VM_DECL_API( AbstrCachePolicy )
public:
	AbstrCachePolicy__pImpl( AbstrCachePolicy *api ) :
	  q_ptr( api ) {}
	AbstrMemoryCache* ownerCache = nullptr;
};

AbstrCachePolicy::AbstrCachePolicy( ::vm::IRefCnt *cnt ) :
  AbstrMemoryCache( cnt ), d_ptr( new AbstrCachePolicy__pImpl( this ) )
{
}

AbstrMemoryCache *AbstrCachePolicy::GetOwnerCache()
{
	VM_IMPL( AbstrCachePolicy )
	return _->ownerCache;
}

inline const void *AbstrCachePolicy::GetPage( size_t pageID )
{
	return nullptr;
}

inline size_t AbstrCachePolicy::GetPageSize() const
{
	return 0;
}

inline size_t AbstrCachePolicy::GetPhysicalPageCount() const
{
	return 0;
}

inline size_t AbstrCachePolicy::GetVirtualPageCount() const
{
	return 0;
}


AbstrCachePolicy::~AbstrCachePolicy()
{
}

void AbstrCachePolicy::SetOwnerCache( AbstrMemoryCache *cache )
{
	VM_IMPL( AbstrCachePolicy )
	_->ownerCache = cache;
}
void AbstrCachePolicy::Invoke_Replace_Event( size_t evictPageID )
{
	VM_IMPL( AbstrCachePolicy )
    auto cache = _->ownerCache;
    if(cache){
      cache->Replace_Event( evictPageID );
    }
}

}  // namespace vm
