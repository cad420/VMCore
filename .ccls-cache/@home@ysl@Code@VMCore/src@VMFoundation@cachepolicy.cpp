
#include <VMFoundation/cachepolicy.h>
#include <cassert>
#include <list>
#include <map>

namespace vm
{

class ListBasedLRUCachePolicy__pImpl
{
	VM_DECL_API( ListBasedLRUCachePolicy )

public:
	ListBasedLRUCachePolicy__pImpl( ListBasedLRUCachePolicy *api ) :
	  q_ptr( api ) {}

	struct LRUListCell;
	using LRUList = std::list<LRUListCell>;
	using LRUHash = std::map<int, std::list<LRUListCell>::iterator>;
	struct LRUListCell
	{
		size_t storageID;
		LRUHash::iterator hashIter;
		LRUListCell( size_t index, LRUHash::iterator itr ) :
		  storageID{ index },
		  hashIter{ itr } {}
	};
	LRUList m_lruList;
	LRUHash m_blockIdInCache;  // blockId---> (blockIndex,the position of blockIndex in list)
	
};


ListBasedLRUCachePolicy::ListBasedLRUCachePolicy( ::vm::IRefCnt *cnt ) :
  AbstrCachePolicy( cnt ),
  d_ptr( new ListBasedLRUCachePolicy__pImpl( this ) )
{
}

bool ListBasedLRUCachePolicy::QueryPage( size_t pageID )
{
	VM_IMPL( ListBasedLRUCachePolicy )
	return _->m_blockIdInCache.find( pageID ) == _->m_blockIdInCache.end() ? false : true;
}

void ListBasedLRUCachePolicy::UpdatePage( size_t pageID )
{
	VM_IMPL( ListBasedLRUCachePolicy )
	const auto it = _->m_blockIdInCache.find( pageID );
	if ( it == _->m_blockIdInCache.end() ) {
		_->m_lruList.splice( _->m_lruList.begin(), _->m_lruList, it->second );  // move the node that it->second points to the head.
	}
}

size_t ListBasedLRUCachePolicy::QueryAndUpdate( size_t pageID )
{
	VM_IMPL( ListBasedLRUCachePolicy )
	const auto it = _->m_blockIdInCache.find( pageID );
	if ( it == _->m_blockIdInCache.end() ) {
		// replace the last block in cache
		auto &lastCell = _->m_lruList.back();
		_->m_lruList.splice( _->m_lruList.begin(), _->m_lruList, --_->m_lruList.end() );  // move last to head

		const auto newItr = _->m_blockIdInCache.insert( std::make_pair( pageID, _->m_lruList.begin() ) );
		if ( lastCell.hashIter != _->m_blockIdInCache.end() ) {
			_->m_blockIdInCache.erase( lastCell.hashIter );  // Unmapped old
		}
		lastCell.hashIter = newItr.first;  // Mapping new
		return lastCell.storageID;
	} else {
		_->m_lruList.splice( _->m_lruList.begin(), _->m_lruList, it->second );  // move the node that it->second points to the head.
		return it->second->storageID;
	}
}

void *ListBasedLRUCachePolicy::GetRawData()
{
	return nullptr;
}

ListBasedLRUCachePolicy::~ListBasedLRUCachePolicy()
{
	
}

void ListBasedLRUCachePolicy::InitEvent( AbstrMemoryCache *cache )
{
	VM_IMPL( ListBasedLRUCachePolicy )
	assert( cache );
	ListBasedLRUCachePolicy__pImpl::LRUList().swap( _->m_lruList );
	for ( auto i = std::size_t( 0 ); i < cache->GetPhysicalPageCount(); i++ )
		_->m_lruList.push_front( ListBasedLRUCachePolicy__pImpl::LRUListCell( i, _->m_blockIdInCache.end() ) );
}
}  // namespace ysl
