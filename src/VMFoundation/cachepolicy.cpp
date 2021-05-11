#include "VMUtils/common.h"
#include <VMFoundation/cachepolicy.h>
#include <cassert>
#include <VMUtils/fmt.hpp>
#include <VMFoundation/logger.h>
#include <VMFoundation/memorypool.h>
#include <VMUtils/ref.hpp>
#include <cstddef>

namespace vm
{
class ListBasedLRUCachePolicy__pImpl
{
	VM_DECL_API( ListBasedLRUCachePolicy )

public:
	ListBasedLRUCachePolicy__pImpl( ListBasedLRUCachePolicy *api ) :
	  q_ptr( api ) {}

	LRUList m_lruList;
	LRUHash m_blockIdInCache;  // blockId---> (blockIndex, the position of blockIndex in list)
};

ListBasedLRUCachePolicy::ListBasedLRUCachePolicy( ::vm::IRefCnt *cnt ) :
  AbstrCachePolicy( cnt ),
  d_ptr( new ListBasedLRUCachePolicy__pImpl( this ) )
{
}

bool ListBasedLRUCachePolicy::QueryPage( size_t pageID ) const
{
	const auto _ = d_func();
	return _->m_blockIdInCache.find( pageID ) == _->m_blockIdInCache.end() ? false : true;
}

/**
 * \brief Update the policy internal state by the given \a pageID.
 * 
 * For example, If it is a lru policy, after calling the function, it will update the LRU records.
 */
void ListBasedLRUCachePolicy::UpdatePage( size_t pageID )
{
	VM_IMPL( ListBasedLRUCachePolicy )
	const auto it = _->m_blockIdInCache.find( pageID );
	if ( it == _->m_blockIdInCache.end() ) {
		_->m_lruList.splice( _->m_lruList.begin(), _->m_lruList, it->second.pa );  // move the node that it->second points to the head.
	}
}

size_t ListBasedLRUCachePolicy::QueryAndUpdate( size_t pageID )
{
	VM_IMPL( ListBasedLRUCachePolicy )
	const auto it = _->m_blockIdInCache.find( pageID );
	if ( it == _->m_blockIdInCache.end() ) {
		// Replaces the least recently used block (the last one) if cache miss
		// Before replacing, it's necessary to check if the cache is dirty and write back



    ////////
    // Update policy state
		_->m_lruList.splice( _->m_lruList.begin(), _->m_lruList, --_->m_lruList.end() );  // move from rear to head
    ////////

		int pteFlags = PTE::PTE_V;
		const auto newItr = _->m_blockIdInCache.insert( std::make_pair( pageID, PTE{ _->m_lruList.begin(), pteFlags } ) );

		auto &eviction = _->m_lruList.back();
		if ( eviction.pte != _->m_blockIdInCache.end() ) {
			// Unmapped old if the virtual address associate an old one
			_->m_blockIdInCache.erase( eviction.pte );	 // Another way is to set invalid flag to pte to indicate this cache is invalid

			Ref<AbstrMemoryCache> cache = GetOwnerCache();
			if (cache) {
				const auto evictPageID = eviction.pte->first;
				/// TODO::
				Invoke_Replace_Event(evictPageID);
			}
		}
		eviction.pte = newItr.first;  // Mapping new
		return eviction.storageID;
	} else {
		// cache hit
    ////////
    // Update policy state
		_->m_lruList.splice( _->m_lruList.begin(), _->m_lruList, it->second.pa );  // move the node that it->second.pa points to the head.
    ////////
		return it->second.pa->storageID;
	}
}

/**
 * \brief Returns the pte with respect to the virtual address \a pageID
 *
 * */
void *ListBasedLRUCachePolicy::QueryPageEntry( size_t pageID ) const
{
	const auto _ = d_func();
	const auto it = _->m_blockIdInCache.find( pageID );
	if ( it == _->m_blockIdInCache.end() ) {
		return nullptr;
	}
	return (void *)&( it->second );
}

void *ListBasedLRUCachePolicy::GetRawData()
{
	return nullptr;
}

LRUList &ListBasedLRUCachePolicy::GetLRUList()
{
	VM_IMPL( ListBasedLRUCachePolicy )
	return _->m_lruList;
}
LRUHash &ListBasedLRUCachePolicy::GetLRUHash()
{
	VM_IMPL( ListBasedLRUCachePolicy )
	return _->m_blockIdInCache;
}

ListBasedLRUCachePolicy::~ListBasedLRUCachePolicy()
{
}

void ListBasedLRUCachePolicy::InitEvent( AbstrMemoryCache *cache )
{
	VM_IMPL( ListBasedLRUCachePolicy )
	assert( cache );
	LRUList().swap( _->m_lruList );
	for ( auto i = std::size_t( 0 ); i < cache->GetPhysicalPageCount(); i++ )
		_->m_lruList.push_front( LRUListItem( i, _->m_blockIdInCache.end() ) );
}
void ListBasedLRUCachePolicy::Replace_Event( size_t evictPageID )
{
}
////////////////////////////////////
//

class LRUCachePolicy__pImpl
{
	VM_DECL_API( LRUCachePolicy )
public:
	LRUCachePolicy__pImpl( LRUCachePolicy *api ) :
	  q_ptr( api )
	{
		m_pagetable = (pagetable_t)malloc( LEVELSIZE );
	}

	// THE CODE BELLOW IS MAINLY ORIGINAL FROM XV6 AN OPERATING SYSTEM FOR TUTORIAL WITH LIGHTLY MODIFICATION.
	// FOR MORE DETAILS, SEE https://pdos.csail.mit.edu/6.S081/2020/xv6/book-riscv-rev1.pdf

	using pagetable_t = uint64_t *;
	using pte_t = uint64_t;
	static constexpr int NBIT = 64;
	static constexpr int NMAXVABIT = 48;  // This value is OS-dependent, 48 is proper for most OS
	static constexpr int NLEVEL = 4;
	static constexpr int LEVELSIZE = ( 1L << ( NBIT / NLEVEL ) );
	static constexpr int PXMASK = 0xFFFFFF;	 // NBIT/NEVEL bits

	static constexpr int NFLAGBIT = 16;
	static constexpr int FLAGMASK = 0x3FF;	// NFLAGBIT bits

	static constexpr int PGSHIFT = 0;  // we have no page offset in page id (address)
	static constexpr int NLEVELBIT = NBIT / NLEVEL;
	static constexpr uint64_t NMAXVA = 1L << NMAXVABIT;
	static constexpr int NLEVELSIZE = ( 1 << NLEVELBIT );

	static constexpr int PXSHIFT( int level ) { return PGSHIFT + NLEVELBIT * level; }
	static constexpr int PX( int level, int va ) { return ( va >> PXSHIFT( level ) ) & PXMASK; }

	static constexpr uint64_t PTE2VA( pte_t pte ) { return ( ( pte ) >> NFLAGBIT ) << PGSHIFT; }
	static constexpr uint64_t VA2PTE( uint64_t pa ) { return ( ( pa ) >> PGSHIFT ) << NFLAGBIT; }
	static constexpr int PTE_FLAGS( uint64_t a ) { return a & FLAGMASK; }
	static constexpr int PTE_V = 1L << 0;  // valid
	static constexpr int PTE_D = 1L << 1;  // dirty

	pagetable_t m_pagetable = nullptr;

	struct LRUEntry
	{
		LRUEntry( size_t storageID, pte_t *pte ) :
		  storageID( storageID ), pte( pte ) {}
		size_t storageID;
		pte_t *pte = nullptr;  // null indicates unused entry
	};
	using LRU_Recorder = std::list<LRUEntry>;

	LRU_Recorder m_recorder;

	/// <summary>
	/// </summary>
	/// <param name="page_id"> as if it is a virtual address in os virtual memory management</param>
	/// <returns></returns>

	/**
	* \brief Return a entry item which records the page given by \a page_id state in cache.
	*
	* \note 
	*   A 64-bit virtual address is split into five fields:
	*   48..63 -- 16 bits of level-3 index.
	*   32..47 -- 16 bits of level-2 index.
	*   16..31 -- 16 bits of level-1 index.
	*    0..15 -- 16 bits of level-0 index.
	*    0..0 -- 0 bits of byte offset within the page.
	*/

	pte_t *Walk( size_t page_id ) const
	{
		auto pagetable = m_pagetable;
		for ( int level = NLEVEL - 1; level > 0; level-- ) {
			pte_t *pte = &pagetable[ PX( level, page_id ) ];
			if ( *pte & PTE_V ) {
				pagetable = (pagetable_t)PTE2VA( *pte );
			} else {
				if ( ( pagetable = (pte_t *)malloc( NLEVELSIZE ) ) == 0 )
					return nullptr;
				memset( pagetable, 0, NLEVELSIZE );
				*pte = VA2PTE( (uint64_t)pagetable ) | PTE_V;
			}
		}
		return &pagetable[ PX( 0, page_id ) ];
	}

	/**
   * \brief This function is called when allocating a LRU Entry corresponding to the given pte.
   * It records the actual page index (or physical address in terms of OS Memory Management) and 
   * reference the given pte itself. The function acts like allocating a physical page.
   * */
	inline LRUEntry *GetOrAllocLRUEntry( pte_t *pte )
	{
		//TODO:: Append an entry to the list and returns its pointer
		assert( pte && "pte pointer is null in GetOrAllocLRUEntry" );
		if ( pte != nullptr ) {
			auto flags = PTE_FLAGS( *pte );
			if ( flags & PTE_D ) {
			} else {
			}

			return 0;
		} else {
			m_recorder.emplace_front( 10, pte );
			return &( *m_recorder.begin() );
		}
	}

	/**
	*  \brief Return the page actual index (physical index) of the given \a page_id (virtual index)
	*/
	size_t Walkaddr( size_t page_id ) const
	{
		//
		auto pte = Walk( page_id );
		if ( PTE_FLAGS( *pte ) & PTE_V ) {
			//TODO:: Returns the corresponding storageID in LRU_Recorder
			return 0;
		}
	}

	/**
	* Frees the entire pagetable which records the mapping between virtual index and the actual index.
	*/
	void Freewalk( pagetable_t pagetable )
	{
		for ( int i = 0; i < LEVELSIZE; i++ ) {
			pte_t pte = pagetable[ i ];
			if ( ( pte & PTE_V ) == 0 ) {
				// this PTE points to a lower-level page table.
				auto child = PTE2VA( pte );
				Freewalk( (pagetable_t)child );
				pagetable[ i ] = 0;
			} else if ( pte & PTE_V ) {
				LOG_FATAL << "Freewalk";
			}
		}
		free( (void *)pagetable );
	}

	~LRUCachePolicy__pImpl()
	{
		Freewalk( m_pagetable );
	}
};

/**
 * \brief Returns \a true if the page is in cache otherwise returns \a false
 * 
 * \note This function do nothing just 
 */
bool LRUCachePolicy::QueryPage( size_t pageID ) const
{
	auto pte = (const size_t *)this->QueryPageEntry( pageID );
	return ( *pte ) & LRUCachePolicy__pImpl::PTE_V;
}

/**
 * \brief Update the policy internal state by the given \a pageID.
 * 
 * For example, If it is a lru policy, after calling the function, it will update the LRU records.
 */
void LRUCachePolicy::UpdatePage( size_t pageID )
{
}

/**
 * \brief Look up and update the pagetable and returns address (block id ) the last level entry point to
 */
size_t LRUCachePolicy::QueryAndUpdate( size_t pageID )
{
	VM_IMPL( LRUCachePolicy );
	return _->Walkaddr( pageID );
}

/**
 * \brief. Returns the page entry info according to the \param pageID
 */
void *LRUCachePolicy::QueryPageEntry( size_t pageID ) const
{
	auto _ = d_func();
	return _->Walk( pageID );
}

/**
 * .\brief Reserved for further use. Maybe just returns the memory pool used by the page table allocation.
 */

void *LRUCachePolicy::GetRawData()
{
	return nullptr;
}

void LRUCachePolicy::InitEvent( AbstrMemoryCache *cache )
{
	VM_IMPL( LRUCachePolicy )
	assert( cache );
	LRUCachePolicy__pImpl::LRU_Recorder().swap( _->m_recorder );
	for ( auto i = std::size_t( 0 ); i < cache->GetPhysicalPageCount(); i++ )
		_->m_recorder.push_front( LRUCachePolicy__pImpl::LRUEntry( i, nullptr ) );
}

/**
 * \brief
 */
vm::LRUCachePolicy::~LRUCachePolicy()
{
}

}  // namespace vm
