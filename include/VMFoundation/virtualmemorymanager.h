#pragma once

#include <VMat/geometry.h>
#include <VMFoundation/foundation_config.h>

#include <VMCoreExtension/i3dblockfileplugininterface.h>
#include <VMUtils/ieverything.hpp>
#include <VMUtils/common.h>

namespace vm
{
class AbstrCachePolicy;

/**
* \brief This class is used to represent a generic cache abstraction layer.
*
*/

class AbstrMemoryCache;
class AbstrMemoryCache__pImpl;

class VMFOUNDATION_EXPORTS AbstrMemoryCache : public ::vm::EverythingBase<IPageFile>
{
public:
	AbstrMemoryCache( IRefCnt *cnt );

	void SetNextLevelCache( IPageFile *cache );
	/**
		 * \brief Sets a cache policy
		 * \param policy
		 */
	void SetCachePolicy( AbstrCachePolicy *policy );
	AbstrCachePolicy *TakeCachePolicy();

	IPageFile *GetNextLevelCache();
	//void SetPageFaultEventCallback(std::shared_ptr<IPageFaultEventCallback> callback) { this->callback = std::move(callback); }
	const IPageFile *GetNextLevelCache() const;

	/**
		 * \brief Get the page given by \a pageID. If the page does not exist in the cache, it will be swapped in.
		 * \note The page data pointed by the  pointer returned by the function is only valid at current call.
		 * It could be invalid when next call because its data has been swapped out.
		 */
	const void *GetPage( size_t pageID ) override;

	virtual void *GetRawData() = 0;

	void Flush() override;

	void Write( const void *page, size_t pageID, bool flush ) override;

	void Flush( size_t pageID ) override;

	virtual ~AbstrMemoryCache();

protected:
	virtual void *GetPageStorage_Implement( size_t pageID ) = 0;
	virtual void PageSwapIn_Implement(void * currentLevelPage,const void * nextLevelPage);
	virtual void PageSwapOut_Implement(void * nextLevelPage, const void * currentLevel);
	virtual void PageWrite_Implement(void * currentLevelPage, const void * userData);
private:
	/**
		 * \brief
		 * \param pageID
		 * \return
	*/
	VM_DECL_IMPL( AbstrMemoryCache )

	friend class AbstrCachePolicy;
};

class AbstrCachePolicy__pImpl;

using PageFlag = int;

enum PageFlagBits
{
	PAGE_FAULT = 0,
	PAGE_V = 1L << 0,  // valid
	PAGE_D = 1L << 1   // dirty
};
struct PageQuery{
	size_t PageID;
	size_t EvictedPageID;
	size_t StorageID;
	PageFlag PageFlags;
	bool Hit;
	bool Evicted;
};

class VMFOUNDATION_EXPORTS AbstrCachePolicy : public AbstrMemoryCache
{
	VM_DECL_IMPL( AbstrCachePolicy )
public:
	AbstrCachePolicy( ::vm::IRefCnt *cnt );

	/**
	*   \brief Queries the page given by \a pageID if it exists in storage cache. Returns \a true if it exists or \a false if not
	*/
	virtual bool QueryPage( size_t pageID ) const = 0;

	/**
	*   \brief Updates the fault page given by \a pageID. Returns the actually storage ID of the page. If the page exists, the function does nothing.
	*/

	virtual void UpdatePage( size_t pageID ) = 0;

	/**
	* \brief Queries and updates at the same time. It will always return a valid storage id.
	*/

	virtual size_t EndQueryAndUpdate( size_t pageID ) = 0;

	virtual void EndQueryAndUpdate( size_t pageID, bool &hit, size_t *storageID, bool &evicted, size_t *evictedPageID ) = 0;

	virtual void BeginQuery( size_t pageID, bool &hit, bool &evicted, size_t &storageID, size_t &evictedPageID ) = 0;

	virtual void BeginQuery(PageQuery * query) = 0;

	virtual void EndQueryAndUpdate(PageQuery * query) = 0;

	/**
	* \brief Queries the page entry given by \a pageID. It includes the page flags state. The meaning dependes on implementation.
	* Returns null pointer if the page does not exists in cache
	*/
	virtual void QueryPageFlag( size_t pageID, PageFlag **pf ) = 0;

	AbstrMemoryCache *GetOwnerCache();

	const void *GetPage( size_t pageID ) override;

	size_t GetPageSize() const override;

	size_t GetPhysicalPageCount() const override;

	size_t GetVirtualPageCount() const override;

	virtual ~AbstrCachePolicy();

protected:
	void *GetPageStorage_Implement( size_t pageID ) override { return nullptr; }
	virtual void InitEvent( AbstrMemoryCache *cache ) = 0;

private:
	friend AbstrMemoryCache;
	void SetOwnerCache( AbstrMemoryCache *cache );
};

}  // namespace vm
