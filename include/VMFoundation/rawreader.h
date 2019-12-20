
#ifndef _RAWREADER_H_
#define _RAWREADER_H_
#include <VMat/geometry.h>
#include <VMCoreExtension/ifilemappingplugininterface.h>
#include <VMFoundation/foundation_config.h>
#include <VMUtils/ref.hpp>
#include <fstream>
#include <VMCoreExtension/i3dblockfileplugininterface.h>
#include <VMFoundation/dataarena.h>
#include <VMUtils/common.h>

namespace vm
{

class RawReader__pImpl;

class VMFOUNDATION_EXPORTS RawReader
{
	VM_DECL_IMPL( RawReader )
public:
	using PosType = unsigned long long;
	using OffsetType = unsigned long long;
	using SizeType = std::size_t;
	RawReader( const std::string &fileName,
				 const vm::Size3 &dimensions, size_t voxelSize );
	RawReader( const std::string &fileName, const Size3 &dimensions, size_t voxelSize,bool mapped);
	
	~RawReader();
	// Read a region of volume data from the file into the buffer passed.
	// It's assumed the buffer passed has enough room. Returns the
	// number voxels read

	size_t readRegion( const vm::Vec3i &start,
					   const vm::Size3 &size, unsigned char *buffer );
	
	size_t readRegionNoBoundary( const vm::Vec3i &start,
					   const vm::Size3 &size, unsigned char *buffer );
	Vec3i GetDimension() const;
	
	size_t GetElementSize() const;

private:
	std::size_t readRegion__( const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer );
	bool convexRead( const vm::Size3 &size ) const;
};

}  // namespace vm
#endif