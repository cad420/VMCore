#pragma once
#include <VMFoundation/foundation_config.h>
#include <VMFoundation/virtualmemorymanager.h>
#include <list>
#include <map>

namespace vm
{
class ListBasedLRUCachePolicy__pImpl;

struct LRUListCell;
using LRUList = std::list<LRUListCell>;
struct PTE{
  std::list<LRUListCell>::iterator pa_itr;
  int flags;
  PTE(std::list<LRUListCell>::iterator itr, int f):pa_itr(itr),flags(f){}
};
using LRUHash = std::map<size_t, PTE>;
struct LRUListCell
{
	size_t storageID;
	LRUHash::iterator hashIter;
	LRUListCell( size_t index, LRUHash::iterator itr ) :
	  storageID{ index },
	  hashIter{ itr } {}
};

class VMFOUNDATION_EXPORTS ListBasedLRUCachePolicy : public AbstrCachePolicy
{
	VM_DECL_IMPL( ListBasedLRUCachePolicy )
public:
	ListBasedLRUCachePolicy( vm::IRefCnt *cnt );
	bool QueryPage( size_t pageID )const override;
	void UpdatePage( size_t pageID ) override;
	size_t QueryAndUpdate( size_t pageID ) override;
	void *QueryPageEntry( size_t pageID )const override;
	void *GetRawData() override;
	~ListBasedLRUCachePolicy();

	LRUList & GetLRUList();
	LRUHash & GetLRUHash();

protected:
	void InitEvent( AbstrMemoryCache *cache ) override;
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
	bool QueryPage( size_t pageID )const override;
	void UpdatePage( size_t pageID ) override;
	size_t QueryAndUpdate( size_t pageID ) override;
	void* QueryPageEntry( size_t pageID )const override;
	void* GetRawData() override;
	~LRUCachePolicy();
protected:
	void InitEvent( AbstrMemoryCache *cache ) override;
};

}  // namespace vm
