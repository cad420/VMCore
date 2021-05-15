
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

struct Block3DDataFileDesc
{
	int Padding = 2;
	int VoxelByte = 1;
	int BlockSideInLog = 6;
	int BlockDim[ 3 ];
	int DataSize[ 3 ];
	bool IsDataSize = false;
	const char * FileName;
};

class I3DBlockFilePluginInterface : public I3DBlockDataInterface
{
public:
	virtual void Open( const std::string &fileName ) = 0;
	virtual bool Create( const Block3DDataFileDesc *desc ) = 0;
};

DECLARE_PLUGIN_METADATA( I3DBlockFilePluginInterface, "visualman.blockdata.io" )
