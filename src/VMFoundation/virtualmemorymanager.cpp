#include "VMUtils/ieverything.hpp"
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
	std::unordered_map<size_t, size_t> dirtyPageID;
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

//const void *AbstrMemoryCache::GetPage( size_t pageID )
//{
//	VM_IMPL( AbstrMemoryCache )
//	assert( _->cachePolicy );
//	const bool e = _->cachePolicy->QueryPage( pageID );
//	if ( !e ) {
//		//const auto storageID = _->cachePolicy->QueryAndUpdate( pageID );
//		size_t storageID, evictedPageID;
//		bool hit, evicted;
//		_->cachePolicy->EndQueryAndUpdate( pageID, hit, &storageID, evicted, &evictedPageID );
//
//		// Read block from next level to the storage cache
//		const auto storage = GetPageStorage_Implement( storageID );
//
//		memcpy( storage, _->nextLevel->GetPage( pageID ), GetPageSize() );
//		return storage;
//	} else {
//		size_t storageID, evictedPageID;
//		bool hit, evicted;
//		_->cachePolicy->EndQueryAndUpdate( pageID, hit, &storageID, evicted, &evictedPageID );
//		//const auto storageID = _->cachePolicy->QueryAndUpdate( pageID );
//		return GetPageStorage_Implement( storageID );
//	}
//}
const void *AbstrMemoryCache::GetPage( size_t pageID )
{
	VM_IMPL( AbstrMemoryCache )
	assert( _->cachePolicy );
	const bool e = _->cachePolicy->QueryPage( pageID );
	bool hit, evicted;
	size_t storageID, evictedPageID;

	_->cachePolicy->BeginQuery( pageID, hit, evicted, storageID, evictedPageID );
	if ( !hit ) {
		// Read block from next level to the storage cache
		//
		///////////////////////
		const auto storage = GetPageStorage_Implement( storageID );
		if ( evicted ) {
			// If a page is evicted, before replaced, it's necessary to write it into the next level cache if the page is dirty.
			PageFlag *evictedFlags;
			_->cachePolicy->QueryPageFlag( evictedPageID, &evictedFlags );
			if ( evictedFlags == nullptr ) 
			{
				LOG_FATAL << "nullptr: evictedFlags";
			}
			else if ( *evictedFlags & PAGE_D ) {
				_->nextLevel->Write( storage, evictedPageID, true );
			}
		}
		// 两个不太清楚的问题：
		// [1] EndQueryAndUpdate 需要提供一个pageID 的 pte吗, 用来更新状态之后设置这个页的pte。? 暂时用QueryPageFlags代替。

		// [2] 现在还有一个问题，就是从下一级缓存把page拿上来的的时候，需要拿到下一级缓存中的这个页的状态(flags)吗？(目前看是不需要的),
		// swap进来的page 就是clean的。
		///////////////////////
		_->cachePolicy->EndQueryAndUpdate( pageID, hit, &storageID, evicted, &evictedPageID );

		memcpy( storage, _->nextLevel->GetPage( pageID ), GetPageSize() );
		return storage;
	} else {
		_->cachePolicy->EndQueryAndUpdate( pageID, hit, &storageID, evicted, &evictedPageID );
		return GetPageStorage_Implement( storageID );
	}
}

void AbstrMemoryCache::Write( const void *page, size_t pageID, bool flush )
{
	VM_IMPL( AbstrMemoryCache )
	if ( flush ) {
		//read, update and write
		auto cachedPage = const_cast<void *>( GetPage( pageID ) );
		memcpy( cachedPage, page, GetPageSize() );
		_->nextLevel->Write( page, pageID, true );	 // update next level cache
	} else {
		// For lazy writing, we need to access the page cache anyway.
		// But note that the evicted page need to be carefully handled
		// because of the probable replacing when we access the page.
		bool hit, evicted;
		size_t storageID, evictedPageID;

		_->cachePolicy->BeginQuery( pageID, hit, evicted, storageID, evictedPageID );
		const auto storage = GetPageStorage_Implement( storageID );
		if ( evicted ) {
			// If a page is evicted, before replaced, it's necessary to write it into the next level cache if the page is dirty.
			PageFlag *evictedFlags;
			_->cachePolicy->QueryPageFlag( evictedPageID, &evictedFlags );
			if ( evictedFlags == nullptr ) 
			{
				LOG_FATAL << "nullptr: evictedFlags";
			} else if( *evictedFlags & PAGE_D ) {
				_->nextLevel->Write( storage, evictedPageID, flush );
			}
		}

		_->cachePolicy->EndQueryAndUpdate( pageID ,hit, &storageID, evicted, &evictedPageID); 

		// 这里有一个比较严重的问题，如果EndQueryAndUpdate在最后，那么QueryPageFlag在hit == false的情况下为空，因为有可能缓存没有命中，
		//这样是拿不到pageflag的，虽然把这个update放在前面会更新并且得到一个pageID合法的pte的位置，但是如果在多线程情况下在之后的QueryPageFlag中
		//会出问题，这个问题的根源还是没有对一个page相关的东西进行原子操作。即理想情况下，对于一个page的所有操作应该夹在BeginQuery--EndQuery之间才行
		//也就是EndQueryAndUpdate的时候需要设置pte的状态。这里接口没有设计好。之后再改

		memcpy( storage, page, GetPageSize() );
		PageFlag *flags;
		_->cachePolicy->QueryPageFlag( pageID, &flags );
		*flags = PAGE_D | PAGE_V;  // mark dirty
		_->dirtyPageID.insert( { pageID, storageID } ); // record the dirty page for flushing all

		// Update will invalidate evictedPageID 
	}
}

void AbstrMemoryCache::Flush( size_t pageID )
{
	//TODO :: write to the next level cache and flush
	VM_IMPL( AbstrMemoryCache );
	auto it = _->dirtyPageID.find( pageID );
	if ( it != _->dirtyPageID.end() ) {
		const auto storageID = it->second;
		PageFlag *flags;
		_->cachePolicy->QueryPageFlag( pageID, &flags );
		if ( ( *flags & PAGE_D ) ) {
			*flags |= ~PAGE_D;	// clear dirty
			_->nextLevel->Write( GetPageStorage_Implement( storageID ), pageID, true );
		} else {
			LOG_FATAL << "The page " << pageID << " is not dirty, flushing is ignored";
		}
		_->dirtyPageID.erase( it );
	}
}

void AbstrMemoryCache::Flush()
{
	VM_IMPL( AbstrMemoryCache )
	for ( auto it = _->dirtyPageID.begin(); it != _->dirtyPageID.end(); ++it ) {
		const auto storageID = it->second;
		PageFlag *flags;
		_->cachePolicy->QueryPageFlag( it->first, &flags );
		if ( ( *flags & PAGE_D ) ) {
			*flags |= ~PAGE_D;	// clear dirty
			_->nextLevel->Write( GetPageStorage_Implement( storageID ), it->first, true );
		} else {
			LOG_FATAL << "The page " << it->first << " is not dirty, flushing is ignored";
		}
		_->dirtyPageID.erase( it );
	}
	_->nextLevel->Flush();
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
	AbstrMemoryCache *ownerCache = nullptr;
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
	if ( cache ) {
		cache->Replace_Event( evictPageID );
	}
}

}  // namespace vm
