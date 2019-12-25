
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
	virtual const void *GetPage( size_t pageID );

	virtual void * GetRawData() = 0;
	
	virtual ~AbstrMemoryCache();

protected:
	virtual void *GetPageStorage_Implement( size_t pageID ) = 0;
private:
	/**
		 * \brief 
		 * \param pageID 
		 * \return 
	*/
	VM_DECL_IMPL( AbstrMemoryCache )
};

class AbstrCachePolicy__pImpl;

class VMFOUNDATION_EXPORTS AbstrCachePolicy : public AbstrMemoryCache
{
	VM_DECL_IMPL( AbstrCachePolicy )
public:
	AbstrCachePolicy( ::vm::IRefCnt *cnt );
	/**
		 * \brief Queries the page given by \a pageID if it exists in storage cache. Returns \a true if it exists or \a false if not
	*/
	virtual bool QueryPage( size_t pageID ) = 0;

	/**
		 * \brief Updates the fault page given by \a pageID. Returns the actually storage ID of the page. If the page exists, the function does nothing.
	*/

	virtual void UpdatePage( size_t pageID ) = 0;
	
	/**
		 * \brief Queries and updates at the same time. It will always return a valid storage id.
		 */
	
	virtual size_t QueryAndUpdate( size_t pageID ) = 0;

	AbstrMemoryCache *GetOwnerCache();

	const AbstrMemoryCache *GetOwnerCache() const;

	const void *GetPage( size_t pageID ) override { return nullptr; }

	size_t GetPageSize() const override { return 0; }

	size_t GetPhysicalPageCount() const override { return 0; }

	size_t GetVirtualPageCount() const override { return 0; }

	virtual ~AbstrCachePolicy();

protected:
	void *GetPageStorage_Implement( size_t pageID ) override { return nullptr; }
	virtual void InitEvent( AbstrMemoryCache *cache ) = 0;

private:
	friend AbstrMemoryCache;
	void SetOwnerCache( AbstrMemoryCache *cache );
};



}  // namespace vm

