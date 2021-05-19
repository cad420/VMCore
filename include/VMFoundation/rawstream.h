
#pragma once

#include <VMat/geometry.h>
#include <VMFoundation/foundation_config.h>
#include <VMUtils/common.h>

#include <cstddef>
#include <functional>
#include <future>

namespace vm
{

class RawStream__pImpl;

class VMFOUNDATION_EXPORTS RawStream
{
	VM_DECL_IMPL( RawStream )
public:
	using PosType = unsigned long long;
	using OffsetType = unsigned long long;
	using SizeType = std::size_t;
	RawStream( const std::string &fileName,
				 const vm::Size3 &dimensions, size_t voxelSize );

  RawStream(unsigned char * src, const Size3 &dimensions, size_t voxelSize);

	~RawStream();

  bool IsOpened()const;

	size_t ReadRegion( const vm::Vec3i &start,
					   const vm::Size3 &size, unsigned char *buffer );
	
	size_t ReadRegionNoBoundary( const vm::Vec3i &start,
					   const vm::Size3 &size, unsigned char *buffer );

	size_t WriteRegion( const vm::Vec3i &start,
					   const vm::Size3 &size,const unsigned char *buffer );
	
	size_t WriteRegionNoBoundary( const vm::Vec3i &start,
					   const vm::Size3 &size,const unsigned char *buffer );
	
	Vec3i GetDimension() const;
	
	size_t GetElementSize() const;

private:
	std::size_t ReadRegion__Implement( const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer );
	std::size_t WriteRegion__Implement( const vm::Vec3i &start, const vm::Size3 &size,const unsigned char *buffer );
	bool IsConvex( const vm::Size3 &size ) const;
};

}  // namespace vm
