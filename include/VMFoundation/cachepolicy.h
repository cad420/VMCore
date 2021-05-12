#pragma once
#include <VMFoundation/foundation_config.h>
#include <VMFoundation/virtualmemorymanager.h>
#include <list>
#include <map>

namespace vm
{
class ListBasedLRUCachePolicy__pImpl;

struct LRUListItem;
using LRUList = std::list<LRUListItem>;


/**
* PageTableEntry struct
*/
struct PTE
{
	PTE( std::list<LRUListItem>::iterator itr, int f ) :
	  pa( itr ), flags( f ) {}

	std::list<LRUListItem>::iterator pa;                          // point to an item in lru list which stores the phyiscal address
	int flags = 0;                                                // invalid
	static constexpr int PTE_V = 1L << 0;                         // valid
	static constexpr int PTE_D = 1L << 1;                         // dirty
};
using LRUHash = std::map<size_t, PTE>;
struct LRUListItem
{
	size_t storageID;
	LRUHash::iterator pte;
	LRUListItem( size_t index, LRUHash::iterator itr ) :
	  storageID{ index },
	  pte{ itr } {}
};

class VMFOUNDATION_EXPORTS ListBasedLRUCachePolicy : public AbstrCachePolicy
{
	VM_DECL_IMPL( ListBasedLRUCachePolicy )
public:
	ListBasedLRUCachePolicy( vm::IRefCnt *cnt );
	bool QueryPage( size_t pageID ) const override;
	void UpdatePage( size_t pageID ) override;
	size_t QueryAndUpdate( size_t pageID ) override;
	void QueryAndUpdate( size_t pageID, size_t *storageID, size_t *evcitedPageID ) override;
	void QueryPageFlag( size_t pageID, PageFlag * pte) const override;
	void *GetRawData() override;
	~ListBasedLRUCachePolicy();

	LRUList &GetLRUList();
	LRUHash &GetLRUHash();

protected:
	void InitEvent( AbstrMemoryCache *cache ) override;
	void Replace_Event( size_t evictPageID ) override;
};

//
// This policy acts as the address translation of virtual memory on OS,
// it uses a continuious memory to record the mapping
//

class LRUCachePolicy__pImpl;

class VMFOUNDATION_EXPORTS LRUCachePolicy : public AbstrCachePolicy
{
	VM_DECL_IMPL( LRUCachePolicy )
public:
	LRUCachePolicy( vm::IRefCnt *cnt );
	bool QueryPage( size_t pageID ) const override;
	void UpdatePage( size_t pageID ) override;
	size_t QueryAndUpdate( size_t pageID ) override;
	void QueryAndUpdate( size_t pageID, size_t *storageID, size_t *evcitedPageID ) override;

	void QueryPageFlag( size_t pageID, PageFlag * pte) const override;
	void *GetRawData() override;
	~LRUCachePolicy();

protected:
	void InitEvent( AbstrMemoryCache *cache ) override;
};

}  // namespace vm
