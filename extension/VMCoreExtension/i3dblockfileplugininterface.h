
#ifndef _PAGEFILEPLUGININTERFACE_H_
#define _PAGEFILEPLUGININTERFACE_H_
#include <VMat/geometry.h>
#include <../extension/VMCoreExtension/plugindef.h>
#include <../extension/VMCoreExtension/ipagefile.h>

namespace ysl
{


class IPageFaultEventCallback
{
public:
	virtual void OnAfterPageSwapInEvent( IPageFile *cache, void *page, size_t pageID ) = 0;
	virtual void OnBeforePageSwapInEvent( IPageFile *cache, void *page, size_t pageID ) = 0;
	virtual ~IPageFaultEventCallback() = default;
};

class I3DBlockFilePluginInterface : public IPageFile
{
public:
	virtual void Open( const std::string &fileName ) = 0;
	virtual int GetPadding() const = 0;
	virtual Size3 GetDataSizeWithoutPadding() const = 0;
	virtual Size3 Get3DPageSize() const = 0;
	virtual int Get3DPageSizeInLog() const = 0;
	virtual Size3 Get3DPageCount() const = 0;
};


DECLARE_PLUGIN_METADATA( I3DBlockFilePluginInterface, "visualman.blockdata.io" )


}  // namespace ysl

#endif