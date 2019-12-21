
#pragma once

#include <VMat/geometry.h>
#include <VMCoreExtension/i3dblockfileplugininterface.h>
#include "foundation_config.h"
#include <VMUtils/common.h>

namespace vm
{

class RawReader;


class GridVolumeFile__pImpl;

class VMFOUNDATION_EXPORTS GridVolumeFile : public EverythingBase<I3DBlockFilePluginInterface>
{
	VM_DECL_IMPL( GridVolumeFile )
	void Create();

public:
	GridVolumeFile( IRefCnt *cnt, const std::string &fileName, const vm::Size3 &dimensions, size_t voxelSize, int blockDimensionInLog );
	GridVolumeFile( IRefCnt *cnt );
	void Open( const std::string &fileName ) override;

	int GetPadding() const override;

	Size3 GetDataSizeWithoutPadding() const override;
	Size3 Get3DPageSize() const override;
	int Get3DPageSizeInLog() const override;

	Size3 Get3DPageCount() const override;
	const const void *GetPage( size_t pageID ) override;

	Vec3i GetDimension() const;
	size_t GetElementSize() const;

	size_t ReadRegion( const Vec3i &start, const Size3 &size, unsigned char *buffer );
	size_t ReadRegionNoBoundary( const Vec3i &start, const Size3 &size, unsigned char *buffer );
};
}