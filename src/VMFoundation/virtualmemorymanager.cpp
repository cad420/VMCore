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

const void *AbstrMemoryCache::GetPage( size_t pageID )
{
	VM_IMPL( AbstrMemoryCache )
	assert( _->cachePolicy );
	const bool e = _->cachePolicy->QueryPage( pageID );

	PageQuery query;
	query.PageID = pageID;
	_->cachePolicy->BeginQuery(&query);
	if ( !query.Hit ) {
		// Read block from next level to the storage cache
		//
		///////////////////////
		const auto storage = GetPageStorage_Implement( query.StorageID );
		if ( query.Evicted ) {
			// If a page is evicted, before replaced, it's necessary to write it into the next level cache if the page is dirty.
			PageFlag *evictedFlags;
			_->cachePolicy->QueryPageFlag( query.EvictedPageID, &evictedFlags );
			if ( evictedFlags == nullptr ) {
				LOG_FATAL << "nullptr: evictedFlags";
			} else if ( *evictedFlags & PAGE_D ) {
				_->nextLevel->Write( storage, query.EvictedPageID, true );
			}
		}
		// ������̫��������⣺
		// [1] EndQueryAndUpdate ��Ҫ�ṩһ��pageID �� pte��, ��������״̬֮���������ҳ��pte��? ��ʱ��QueryPageFlags���档

		// [2] ���ڻ���һ�����⣬���Ǵ���һ�������page�������ĵ�ʱ����Ҫ�õ���һ�������е����ҳ��״̬(flags)��(Ŀǰ���ǲ���Ҫ��),
		// swap������page ����clean�ġ�
		///////////////////////
		_->cachePolicy->EndQueryAndUpdate(&query);

		PageSwapIn_Implement(storage, _->nextLevel->GetPage(pageID));
		//memcpy( storage, _->nextLevel->GetPage( pageID ), GetPageSize() );
		return storage;
	} else {
		//_->cachePolicy->EndQueryAndUpdate( pageID, hit, &storageID, evicted, &evictedPageID );
		_->cachePolicy->EndQueryAndUpdate(&query);
		return GetPageStorage_Implement( query.StorageID );
	}
}

void AbstrMemoryCache::UnlockPage(size_t pageID){

}

void AbstrMemoryCache::Write( const void *page, size_t pageID, bool flush )
{
	VM_IMPL( AbstrMemoryCache )
	if ( flush ) {
		//read, update and write
		auto cachedPage = const_cast<void *>( GetPage( pageID ) );
		PageSwapOut_Implement(cachedPage, page);
		_->nextLevel->Write( page, pageID, true );	// update next level cache
	} else {
		// For lazy writing, we need to access the page cache anyway.
		// But note that the evicted page need to be carefully handled
		// because of the probable replacing when we access the page.

		PageQuery query;
		query.PageID = pageID;
		_->cachePolicy->BeginQuery(&query);
		const auto storage = GetPageStorage_Implement( query.StorageID );
		if ( query.Evicted ) {
			// If a page is evicted, before replaced, it's necessary to write it into the next level cache if the page is dirty.
			PageFlag *evictedFlags;
			_->cachePolicy->QueryPageFlag( query.EvictedPageID, &evictedFlags );
			if ( evictedFlags == nullptr ) {
				LOG_FATAL << "nullptr: evictedFlags";
			} else if ( *evictedFlags & PAGE_D ) {
				_->nextLevel->Write( storage, query.EvictedPageID, flush );
			}
		}

		_->cachePolicy->EndQueryAndUpdate(&query);

		// ������һ���Ƚ����ص����⣬���EndQueryAndUpdate�������ôQueryPageFlag��hit == false�������Ϊ�գ���Ϊ�п��ܻ���û�����У�
		//�������ò���pageflag�ģ���Ȼ�����update����ǰ�����²��ҵõ�һ��pageID�Ϸ���pte��λ�ã���������ڶ��߳��������֮���QueryPageFlag��
		//������⣬�������ĸ�Դ����û�ж�һ��page��صĶ�������ԭ�Ӳ���������������£�����һ��page�����в���Ӧ�ü���BeginQuery--EndQuery֮�����
		//Ҳ����EndQueryAndUpdate��ʱ����Ҫ����pte��״̬������ӿ�û����ƺá�֮���ٸ�

		PageWrite_Implement(storage, page);
		PageFlag *flags;
		_->cachePolicy->QueryPageFlag( pageID, &flags );
		*flags = PAGE_D | PAGE_V;						 // mark dirty
		_->dirtyPageID.insert( { pageID, query.StorageID } );	 // record the dirty page for flushing all
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
		if ( flags ) {
			if ( ( *flags & PAGE_D ) ) {
				*flags |= ~PAGE_D;	// clear dirty
				_->nextLevel->Write( GetPageStorage_Implement( storageID ), pageID, true );
			} else {
				LOG_FATAL << "The page " << pageID << " is not dirty, flushing is ignored";
			}
		}
		_->dirtyPageID.erase( it );
	}
}

void AbstrMemoryCache::PageSwapIn_Implement(void * currentLevelPage,const void * nextLevelPage){
	memcpy(currentLevelPage, nextLevelPage, GetPageSize());
}

void AbstrMemoryCache::PageSwapOut_Implement(void * nextLevelPage, const void * currentLevelPage){
	memcpy(nextLevelPage, currentLevelPage, GetPageSize());
}

void AbstrMemoryCache::PageWrite_Implement(void * currentLevelPage, const void * userData){
	memcpy(currentLevelPage, userData, GetPageSize());
}

void AbstrMemoryCache::Flush()
{
	VM_IMPL( AbstrMemoryCache )
	for ( auto it = _->dirtyPageID.begin(); it != _->dirtyPageID.end(); ++it ) {
		const auto storageID = it->second;
		PageFlag *flags;
		_->cachePolicy->QueryPageFlag( it->first, &flags );
		if ( flags ) {
			if ( ( *flags & PAGE_D ) ) {
				*flags |= ~PAGE_D;	// clear dirty
				_->nextLevel->Write( GetPageStorage_Implement( storageID ), it->first, true );
			} else {
				LOG_FATAL << "The page " << it->first << " is not dirty, flushing is ignored";
			}
		}
	}
	_->dirtyPageID.clear();
	_->nextLevel->Flush();
}

AbstrMemoryCache::~AbstrMemoryCache()
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

inline void AbstrCachePolicy::UnlockPage(size_t pageID){

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

}  // namespace vm
