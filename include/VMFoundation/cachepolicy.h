
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
	 ListBasedLRUCachePolicy( ::vm::IRefCnt *cnt );
	bool QueryPage( size_t pageID ) override;
	void UpdatePage( size_t pageID ) override;
	size_t QueryAndUpdate( size_t pageID ) override;
    void *GetRawData() override;
	~ListBasedLRUCachePolicy();

protected:
	void InitEvent( AbstrMemoryCache *cache ) override;
};





}  // namespace vm
