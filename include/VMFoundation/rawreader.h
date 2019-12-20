
#ifndef _RAWREADER_H_
#define _RAWREADER_H_
#include <VMat/geometry.h>
#include <VMCoreExtension/ifilemappingplugininterface.h>
#include <VMFoundation/foundation_config.h>
#include <VMUtils/ref.hpp>
#include <fstream>
#include <VMCoreExtension/i3dblockfileplugininterface.h>
#include "dataarena.h"

namespace vm
{
class VMFOUNDATION_EXPORTS RawReader
{
	std::string fileName;
	vm::Size3 dimensions;
	size_t voxelSize;
	uint64_t offset;
	//std::shared_ptr<IFileMappingPluginInterface> io;
	::vm::Ref<IFileMapping> io;
	unsigned char *ptr;
	uint64_t seekAmt;



	

	

public:
	using PosType = unsigned long long;
	using OffsetType = unsigned long long;
	using SizeType = std::size_t;

	RawReader( const std::string &fileName,
			   const vm::Size3 &dimensions, size_t voxelSize );
	~RawReader();
	// Read a region of volume data from the file into the buffer passed.
	// It's assumed the buffer passed has enough room. Returns the
	// number voxels read

	size_t readRegion( const vm::Size3 &start,
					   const vm::Size3 &size, unsigned char *buffer );

private:
	std::size_t readRegion__( const vm::Size3 &start, const vm::Size3 &size, unsigned char *buffer );
	bool convexRead( const vm::Size3 &size )
	{
		/// A minimum continuous unit for reading

		// 3 cases for convex reads:
		// - We're reading a set of slices of the volume
		// - We're reading a set of scanlines of a slice
		// - We're reading a set of voxels in a scanline
		return ( size.x == dimensions.x && size.y == dimensions.y ) || ( size.x == dimensions.x && size.z == 1 ) || ( size.y == 1 && size.z == 1 );
	}
};

class VMFOUNDATION_EXPORTS RawReaderIO
{
	std::string fileName;
	vm::Size3 dimensions;
	size_t voxelSize;
	uint64_t offset;
	std::ifstream file;
	unsigned char *ptr;
	
	uint64_t seekAmt;
	uint64_t totalBytes;
	
	DataArena<64> stagingBuffer;
	


	Vec3i stagingBufferSize;
	Vec3i readingBufferSize;

	
	//Bound3i bound;
public:
	using PosType = unsigned long long;
	using OffsetType = unsigned long long;
	using SizeType = std::size_t;

	RawReaderIO( const std::string &fileName,
				 const vm::Size3 &dimensions, size_t voxelSize );
	~RawReaderIO();
	// Read a region of volume data from the file into the buffer passed.
	// It's assumed the buffer passed has enough room. Returns the
	// number voxels read

	size_t readRegion( const vm::Vec3i &start,
					   const vm::Size3 &size, unsigned char *buffer );

	
	size_t readRegionNoBoundary( const vm::Vec3i &start,
					   const vm::Size3 &size, unsigned char *buffer );
	

	Size3 GetDimension() const
	{
		return dimensions;
	}
	size_t GetElementSize() const { return voxelSize; }

	
	
private:
	size_t Transport3D( const unsigned char *src, const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer );
	std::size_t readRegion__( const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer );
	bool convexRead( const vm::Size3 &size )const
	{
		/// A minimum continuous unit for reading

		// 3 cases for convex reads:
		// - We're reading a set of slices of the volume
		// - We're reading a set of scanlines of a slice
		// - We're reading a set of voxels in a scanline
		return ( size.x == dimensions.x && size.y == dimensions.y )
		|| ( size.x == dimensions.x && size.z == 1 )
		|| ( size.y == 1 && size.z == 1 );
	}
	

	//size_t transport3d__( const unsigned char *src, const vm::Vec3i &start, const vm::Size3 &size, unsigned char *dst );
	
	//bool inside(const ysl::Vec3i& start, const ysl::Size3 & size)const
	//{
	//	const Bound3i t(Point3i(start.x, start.y, start.z), Point3i{start.x+size.x,start.y+size.y,start.z+size.z});
	//	return bound.InsideEx(t);
	//}
};


class RawFile : public EverythingBase<I3DBlockFilePluginInterface>
{
	std::unique_ptr<RawReaderIO> rawReader;
	Size3 blockDimension;
	int blockSizeInLog = -1;
	Size3 pageCount;
	const int padding = 0;
	bool exact = false;
	void Create();
	std::unique_ptr<char[]> buf;  // buffer for a block
public:
	RawFile( IRefCnt *cnt, const std::string &fileName, const vm::Size3 &dimensions, size_t voxelSize ,int blockDimensionInLog);
	RawFile( IRefCnt *cnt );
	void Open( const std::string &fileName ) override;
	int GetPadding() const override;

	Size3 GetDataSizeWithoutPadding() const override;
	Size3 Get3DPageSize() const override;
	int Get3DPageSizeInLog() const override;

	Size3 Get3DPageCount() const override;
	const const void *GetPage(size_t pageID) override;

	Size3 GetDimension() const;
	size_t GetElementSize() const;
	size_t ReadRegion( const Vec3i &start, const Size3 &size, unsigned char *buffer );
	size_t ReadRegionNoBoundary( const Vec3i &start, const Size3 &size, unsigned char *buffer );
};

}  // namespace vm
#endif