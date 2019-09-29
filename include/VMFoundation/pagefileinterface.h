
#ifndef _PAGEFILEPLUGININTERFACE_H_
#define _PAGEFILEPLUGININTERFACE_H_
#include <VMat/geometry.h>
#include <../interface/VMCore/plugindef.h>
#include <VMFoundation/object.h>
namespace ysl
{
	//class AbstrMemoryCache;

class IPageFile
{
public:
	virtual ~IPageFile() = default;
	/**
				 * \brief Get the page give by \a pageID. If the page does not exist in the cache, it will be swapped in.
				 * \note The page data pointed by the  pointer returned by the function is only valid at current call.
				 * It could be invalid when next call because its data has been swapped out.
				 */
	virtual const void *GetPage( size_t pageID ) = 0;

	/**
		 * \brief Returns the page size by bytes
		 */
	virtual size_t GetPageSize() const = 0;
	/**
		 * \brief
		 * \return
		 */
	virtual size_t GetPhysicalPageCount() const = 0;
	/**
		 * \brief
		 * \return
		 */
	virtual size_t GetVirtualPageCount() const = 0;

protected:
};

class IPageFaultEventCallback
{
public:
	virtual void OnAfterPageSwapInEvent( IPageFile *cache, void *page, size_t pageID ) = 0;
	virtual void OnBeforePageSwapInEvent( IPageFile *cache, void *page, size_t pageID ) = 0;
	virtual ~IPageFaultEventCallback() = default;
};


	class I3DBlockFilePluginInterface:public Object ,public IPageFile
	{
	public:
		virtual void Open( const std::string &fileName ) = 0;
		virtual int GetPadding() const = 0;
		virtual Size3 GetDataSizeWithoutPadding() const = 0;
		virtual Size3 Get3DPageSize() const = 0;
		virtual int Get3DPageSizeInLog() const = 0;
		virtual Size3 Get3DPageCount() const = 0;
	};

	DECLARE_PLUGIN_METADATA(I3DBlockFilePluginInterface,"visualman.blockdata.io")

   //class IPageFaultEventCallbackPluginInterface:public IPageFaultEventCallback
   //{
   //};

}

#endif