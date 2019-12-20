
#pragma once

#include <VMat/geometry.h>
#include <VMCoreExtension/i3dblockfileplugininterface.h>

namespace vm
{

class RawReader;

class VolumeFile : public EverythingBase<I3DBlockFilePluginInterface>
{
	std::unique_ptr<RawReader> rawReader;
	Size3 blockDimension;
	int blockSizeInLog = -1;
	Size3 pageCount;
	const int padding = 0;
	bool exact = false;
	void Create();
	std::unique_ptr<char[]> buf;  // buffer for a block
public:
	VolumeFile( IRefCnt *cnt, const std::string &fileName, const vm::Size3 &dimensions, size_t voxelSize, int blockDimensionInLog );
	VolumeFile( IRefCnt *cnt );
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