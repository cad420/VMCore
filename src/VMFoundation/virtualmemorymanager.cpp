
#include <cstring>
#include <VMFoundation/virtualmemorymanager.h>

namespace vm
{

class AbstrMemoryCache__pImpl
{
	//AbstrMemoryCache * const q_ptr = nullptr;
	VM_DECL_API( AbstrMemoryCache )
public:
	AbstrMemoryCache__pImpl( AbstrMemoryCache *api ):
	  q_ptr( api ) {}
	Ref<IPageFile> nextLevel;
	Ref<AbstrCachePolicy> cachePolicy;
};

class AbstrCachePolicy__pImpl
{
	VM_DECL_API( AbstrCachePolicy )
public:
	AbstrCachePolicy__pImpl( AbstrCachePolicy *api ) :
	  q_ptr( api ) {}
	AbstrMemoryCache * ownerCache = nullptr;
};


AbstrMemoryCache::AbstrMemoryCache( IRefCnt *cnt ):
	::vm::EverythingBase<IPageFile>( cnt ),d_ptr( new AbstrMemoryCache__pImpl(this) )
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

const IPageFile * AbstrMemoryCache::GetNextLevelCache() const
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
		const auto storageID = _->cachePolicy->QueryAndUpdate( pageID );
		// Read block from next level to the storage cache
		const auto storage = GetPageStorage_Implement( storageID );

		memcpy( storage, _->nextLevel->GetPage( pageID ), GetPageSize() );
		return storage;
	} else {
		const auto storageID = _->cachePolicy->QueryAndUpdate( pageID );
		return GetPageStorage_Implement( storageID );
	}
}

	AbstrMemoryCache::~AbstrMemoryCache()
{

}

AbstrCachePolicy::AbstrCachePolicy( ::vm::IRefCnt *cnt ):
	AbstrMemoryCache( cnt ),d_ptr( new AbstrCachePolicy__pImpl(this) )
{

}

AbstrMemoryCache *AbstrCachePolicy::GetOwnerCache()
{
	VM_IMPL( AbstrCachePolicy )
	return _->ownerCache;
}

const AbstrMemoryCache *AbstrCachePolicy::GetOwnerCache() const
{
	//VM_IMPL( AbstrCachePolicy )
	const auto _ = d_func();
	return _->ownerCache;
}

	AbstrCachePolicy::~AbstrCachePolicy()
{
}

void AbstrCachePolicy::SetOwnerCache( AbstrMemoryCache *cache )
{
	VM_IMPL( AbstrCachePolicy )
	_->ownerCache = cache;
}

}  // namespace ysl
