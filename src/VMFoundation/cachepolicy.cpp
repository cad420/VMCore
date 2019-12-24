
#include <VMFoundation/cachepolicy.h>
#include <cassert>

namespace vm
{

class LRUCachePolicy__pImpl
{
	VM_DECL_API( LRUCachePolicy )

public:
	LRUCachePolicy__pImpl( LRUCachePolicy *api ) :
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


LRUCachePolicy::LRUCachePolicy( ::vm::IRefCnt *cnt ):
	AbstrCachePolicy( cnt ),d_ptr( new LRUCachePolicy__pImpl(this) )
{
}

bool LRUCachePolicy::QueryPage( size_t pageID )
{
	VM_IMPL( LRUCachePolicy )
	return _->m_blockIdInCache.find( pageID ) == _->m_blockIdInCache.end() ? false : true;
}

void LRUCachePolicy::UpdatePage( size_t pageID )
{
	VM_IMPL( LRUCachePolicy )
	const auto it = _->m_blockIdInCache.find( pageID );
	if ( it == _->m_blockIdInCache.end() ) {
		_->m_lruList.splice( _->m_lruList.begin(), _->m_lruList, it->second );  // move the node that it->second points to the head.
	}
}

size_t LRUCachePolicy::QueryAndUpdate( size_t pageID )
{
	VM_IMPL( LRUCachePolicy )
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

	LRUCachePolicy::~LRUCachePolicy()
{
}

void LRUCachePolicy::InitEvent( AbstrMemoryCache *cache )
{
	VM_IMPL( LRUCachePolicy )
	assert( cache );
	LRUCachePolicy__pImpl::LRUList().swap( _->m_lruList );
	for ( auto i = std::size_t( 0 ); i < cache->GetPhysicalPageCount(); i++ )
		_->m_lruList.push_front( LRUCachePolicy__pImpl::LRUListCell( i, _->m_blockIdInCache.end() ) );
}
}  // namespace ysl
