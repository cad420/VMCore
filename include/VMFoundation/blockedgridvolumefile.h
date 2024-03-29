
#pragma once

#include "foundation_config.h"

#include <VMat/geometry.h>
#include <VMUtils/common.h>

#include <VMCoreExtension/i3dblockfileplugininterface.h>
#include <VMCoreExtension/plugin.h>

namespace vm
{

class RawReader;

class BlockedGridVolumeFile__pImpl;

class VMFOUNDATION_EXPORTS BlockedGridVolumeFile : public EverythingBase<I3DBlockFilePluginInterface>
{
	VM_DECL_IMPL( BlockedGridVolumeFile )
public:
	BlockedGridVolumeFile( IRefCnt *cnt, const std::string &fileName, const vm::Size3 &dimensions, size_t voxelSize, int blockDimensionInLog, int padding );
	BlockedGridVolumeFile( IRefCnt *cnt );

	void Open( const std::string &fileName ) override;
	bool Create( const Block3DDataFileDesc *desc ) override;
	void Close() override;
	int GetPadding() const override;
	Size3 GetDataSizeWithoutPadding() const override;
	Size3 Get3DPageSize() const override;
	int Get3DPageSizeInLog() const override;
	Size3 Get3DPageCount() const override;

	size_t GetPhysicalPageCount() const override;
	size_t GetVirtualPageCount() const override;
	size_t GetPageSize() const override;
	const void *GetPage( size_t pageID ) override;

	void Flush() override;

	void Write( const void *page, size_t pageID, bool flush ) override;

	void Flush( size_t pageID ) override;

	Vec3i GetDimension() const;
	size_t GetElementSize() const;
	size_t ReadRegion( const Vec3i &start, const Size3 &size, unsigned char *buffer );
	size_t ReadRegionNoBoundary( const Vec3i &start, const Size3 &size, unsigned char *buffer );

	~BlockedGridVolumeFile();
};

class BlockedGridVolumeFilePluginFactory : public IPluginFactory
{
public:
	DECLARE_PLUGIN_FACTORY( "visualman.blockdata.io" );
	int Keys( const char **keys ) const override;
	IEverything *Create( const char *key ) override;
};

VM_REGISTER_PLUGIN_FACTORY_DECL( BlockedGridVolumeFilePluginFactory )

}  // namespace vm
