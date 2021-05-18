
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
	int Padding = 2;          // Padding of a block
	int VoxelByte = 1;        // The voxel size of the data
	int BlockSideInLog = 6;   // The side length of a block
	int BlockDim[ 3 ];        // How many block with respect to each dimension the data has
	int DataSize[ 3 ];        // Data size
	bool IsDataSize = false;  // True indicates the data create by \ref DataSize or by \ref BlockDim and BlockDim
	const char * FileName = nullptr;
};

class I3DBlockFilePluginInterface : public I3DBlockDataInterface
{
public:
	virtual void Open( const std::string &fileName ) = 0;
	virtual bool Create( const Block3DDataFileDesc *desc ) = 0;
	virtual void Close() = 0;
};

DECLARE_PLUGIN_METADATA( I3DBlockFilePluginInterface, "visualman.blockdata.io" )
