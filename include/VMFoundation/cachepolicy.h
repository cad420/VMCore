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
	bool QueryPage( size_t pageID ) const override;
	void QueryPageFlag( size_t pageID, PageFlag ** pf) override;
	void BeginQuery(PageQuery * query) override;
	void EndQueryAndUpdate(PageQuery * query) override;
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
	bool QueryPage( size_t pageID ) const override;
	void QueryPageFlag( size_t pageID, PageFlag ** pf) override;
	void BeginQuery(PageQuery * query) override;
	void *GetRawData() override;
	~LRUCachePolicy();

protected:
	void InitEvent( AbstrMemoryCache *cache ) override;
};

}  // namespace vm
