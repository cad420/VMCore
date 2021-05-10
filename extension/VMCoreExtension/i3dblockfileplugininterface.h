
#pragma once

#include <VMat/geometry.h>
#include <../extension/VMCoreExtension/plugindef.h>
#include <../extension/VMCoreExtension/ipagefile.h>

class IPageFaultEventCallback
{
public:
	virtual void OnAfterPageSwapInEvent( vm::IPageFile *cache, void *page, size_t pageID ) = 0;
	virtual void OnBeforePageSwapInEvent( vm::IPageFile *cache, void *page, size_t pageID ) = 0;
	virtual ~IPageFaultEventCallback() = default;
};

class I3DBlockDataInterface : public vm::IPageFile
{
public:
	virtual int GetPadding() const = 0;
	virtual vm::Size3 GetDataSizeWithoutPadding() const = 0;
	virtual vm::Size3 Get3DPageSize() const = 0;
	virtual int Get3DPageSizeInLog() const = 0;
	virtual vm::Size3 Get3DPageCount() const = 0;
};

class I3DBlockFilePluginInterface : public I3DBlockDataInterface
{
public:
	virtual void Open( const std::string &fileName ) = 0;
};

DECLARE_PLUGIN_METADATA( I3DBlockFilePluginInterface, "visualman.blockdata.io" )
