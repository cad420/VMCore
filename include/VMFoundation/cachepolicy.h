
#pragma once

#include <VMFoundation/foundation_config.h>
#include <VMFoundation/virtualmemorymanager.h>

namespace vm
{
class ListBasedLRUCachePolicy__pImpl;

class VMFOUNDATION_EXPORTS ListBasedLRUCachePolicy : public AbstrCachePolicy
{
	VM_DECL_IMPL( ListBasedLRUCachePolicy )
public:
	ListBasedLRUCachePolicy( vm::IRefCnt *cnt );
	bool QueryPage( size_t pageID )const override;
	void UpdatePage( size_t pageID ) override;
	size_t QueryAndUpdate( size_t pageID ) override;
	size_t QueryPageEntry( size_t pageID )const override;
	void *GetRawData() override;
	~ListBasedLRUCachePolicy();

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
	size_t QueryPageEntry( size_t pageID )const override;
	void *GetRawData() override;
	~LRUCachePolicy();
};

}  // namespace vm
