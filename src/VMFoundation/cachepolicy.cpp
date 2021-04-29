
#include <VMFoundation/cachepolicy.h>
#include <cassert>
#include <list>
#include <map>
#include <VMFoundation/logger.h>

//#define PGSIZE 4096	 // bytes per page
//#define PGSHIFT 12	 // bits of offset within a page
//
//#define PGROUNDUP( sz ) ( ( ( sz ) + PGSIZE - 1 ) & ~( PGSIZE - 1 ) )
//#define PGROUNDDOWN( a ) ( ( ( a ) ) & ~( PGSIZE - 1 ) )
//
//#define PTE_V ( 1L << 0 )  // valid
//#define PTE_R ( 1L << 1 )
//#define PTE_W ( 1L << 2 )
//#define PTE_X ( 1L << 3 )
//#define PTE_U ( 1L << 4 )  // 1 -> user can access
//
//// shift a physical address to the right place for a PTE.
//#define PA2PTE( pa ) ( ( ( (uint64)pa ) >> 12 ) << 10 )
//
//#define PTE2PA( pte ) ( ( ( pte ) >> 10 ) << 12 )
//
//#define PTE_FLAGS( pte ) ( (pte)&0x3FF )
//
//// extract the three 9-bit page table indices from a virtual address.
//#define PXMASK 0x1FF  // 9 bits
//#define PXSHIFT( level ) ( PGSHIFT + ( 9 * ( level ) ) )
//#define PX( level, va ) ( ( ( ( uint64 )( va ) ) >> PXSHIFT( level ) ) & PXMASK )

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
		_->m_lruList.splice( _->m_lruList.begin(), _->m_lruList, it->second );	// move the node that it->second points to the head.
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
			_->m_blockIdInCache.erase( lastCell.hashIter );	 // Unmapped old
		}
		lastCell.hashIter = newItr.first;  // Mapping new
		return lastCell.storageID;
	} else {
		_->m_lruList.splice( _->m_lruList.begin(), _->m_lruList, it->second );	// move the node that it->second points to the head.
		return it->second->storageID;
	}
}

size_t ListBasedLRUCachePolicy::QueryPageEntry( size_t pageID )
{
	return 0;
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
////////////////////////////////////

class LRUCachePolicy__pImpl
{
	VM_DECL_API( LRUCachePolicy )
public:
	LRUCachePolicy__pImpl( LRUCachePolicy *api ) :
	  q_ptr( api ) {
		m_pagetable = (pagetable_t)malloc( LEVELSIZE );
	}


	// THE CODE BELLOW IS MAINLY ORIGINAL FROM XV6 AN OPERATING SYSTEM FOR TUTORIAL WITH LIGHTLY MODIFICATION.
	// FOR MORE DETAILS, SEE https://pdos.csail.mit.edu/6.S081/2020/xv6/book-riscv-rev1.pdf 

	using pagetable_t = uint64_t *;
	using pte_t = uint64_t;
	static constexpr int NBIT = 64;
	static constexpr int NLEVEL = 4;
	static constexpr int LEVELSIZE = ( 1L << ( NBIT / NLEVEL ) );
	static constexpr int PXMASK = 0xFFFFFF;	 // NBIT/NEVEL bits

	static constexpr int NFLAGBIT = 10;
	static constexpr int FLAGMASK = 0x3FF;	// NFLAGBIT bits

	static constexpr int PGSHIFT = 0;  // we have no page offset in page id (address)
	static constexpr int NLEVELBIT = NBIT / NLEVEL;
	static constexpr int NLEVELSIZE = ( 1 << NLEVELBIT );

	static constexpr int PXSHIFT( int level ) { return PGSHIFT + LEVELSIZE * level; }
	static constexpr int PX( int level, int va ) { return ( va >> PXSHIFT( level ) ) & PXMASK; }

	static constexpr uint64_t PTE2PA( pte_t pte ) { return ( ( pte ) >> NFLAGBIT ) << PGSHIFT; }
	static constexpr uint64_t PA2PTE( uint64_t pa ) { return ( ( pa ) >> PGSHIFT ) << NFLAGBIT; }
	static constexpr int PTE_FLAGS( uint64_t a ) { return a & FLAGMASK; }
	static constexpr int PTE_V = 1L << 0;  // valid
	static constexpr int PTE_D = 1L << 1;  // dirty

	pagetable_t m_pagetable = nullptr;

	/// <summary>
	/// Return the address of the PTE in page table pagetable
	/// that corresponds to virtual address page_id.
	/// create any required page-table pages.
	///
	/// A 64-bit virtual address is split into five fields:
	///   48..63 -- 16 bits of level-3 index.
	///   32..47 -- 16 bits of level-2 index.
	///   16..31 -- 16 bits of level-1 index.
	///    0..15 -- 16 bits of level-0 index.
	///    0..0 -- 0 bits of byte offset within the page.
	/// </summary>
	/// <param name="page_id"> as if it is a virtual address in os virtual memory management</param>
	/// <returns></returns>

	pte_t *Walk( size_t page_id )
	{
		auto pagetable = m_pagetable;
		for ( int level = NLEVEL - 1; level > 0; level-- ) {
			pte_t *pte = &pagetable[ PX( level, page_id ) ];
			if ( *pte & PTE_V ) {
				pagetable = (pagetable_t)PTE2PA( *pte );
			} else {
				if ( ( pagetable = (pte_t *)malloc( NLEVELSIZE ) ) == 0 )
					return 0;
				memset( pagetable, 0, NLEVELSIZE );
				*pte = PA2PTE( (uint64_t)pagetable ) | PTE_V;
			}
		}
		return &pagetable[ PX( 0, page_id ) ];
	}

	void Freewalk( pagetable_t pagetable )
	{
		for ( int i = 0; i < LEVELSIZE; i++ ) {
			pte_t pte = pagetable[ i ];
			if ( ( pte & PTE_V ) == 0 ) {
				// this PTE points to a lower-level page table.
				auto child = PTE2PA( pte );
				Freewalk( (pagetable_t)child );
				pagetable[ i ] = 0;
			} else if ( pte & PTE_V ) {
				LOG_FATAL<<"Freewalk";
			}
		}
		free( (void *)pagetable );
	}

	~LRUCachePolicy__pImpl() {
		Freewalk( m_pagetable );
	}
};

/// <summary>
/// LRUCachePolicy
/// </summary>
/// <param name="pageID"></param>
/// <returns></returns>

bool LRUCachePolicy::QueryPage( size_t pageID )
{
	return false;
}

/// <summary>
///
/// </summary>
/// <param name="pageID"></param>
void LRUCachePolicy::UpdatePage( size_t pageID )
{
}

/// <summary>
///
/// </summary>
/// <param name="pageID"></param>
/// <returns></returns>
size_t LRUCachePolicy::QueryAndUpdate( size_t pageID )
{
	return size_t();
}

/// <summary>
///
/// </summary>
/// <param name="pageID"></param>
/// <returns></returns>
size_t LRUCachePolicy::QueryPageEntry( size_t pageID )
{
	return 0;
}

/// <summary>
///
/// </summary>
/// <returns></returns>
void *LRUCachePolicy::GetRawData()
{
	return nullptr;
}

vm::LRUCachePolicy::~LRUCachePolicy()
{
}

}  // namespace vm
