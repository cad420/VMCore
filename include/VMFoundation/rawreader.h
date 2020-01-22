
#ifndef _RAWREADER_H_
#define _RAWREADER_H_
#include <VMat/geometry.h>
#include <VMCoreExtension/ifilemappingplugininterface.h>
#include <VMFoundation/foundation_config.h>
#include <VMUtils/ref.hpp>
#include <fstream>

namespace vm
{
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

private:
	std::size_t readRegion__( const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer );
	bool convexRead( const vm::Size3 &size )
	{
		/// A minimum continuous unit for reading

		// 3 cases for convex reads:
		// - We're reading a set of slices of the volume
		// - We're reading a set of scanlines of a slice
		// - We're reading a set of voxels in a scanline
		return ( size.x == dimensions.x && size.y == dimensions.y ) || ( size.x == dimensions.x && size.z == 1 ) || ( size.y == 1 && size.z == 1 );
	}
	//bool inside(const ysl::Vec3i& start, const ysl::Size3 & size)const
	//{
	//	const Bound3i t(Point3i(start.x, start.y, start.z), Point3i{start.x+size.x,start.y+size.y,start.z+size.z});
	//	return bound.InsideEx(t);
	//}
};

}  // namespace vm
#endif