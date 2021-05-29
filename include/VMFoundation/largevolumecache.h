#pragma once
#include "lvdreader.h"
#include <VMat/geometry.h>
#include <VMFoundation/virtualmemorymanager.h>
#include <VMFoundation/genericcache.h>
#include <VMUtils/ieverything.hpp>
#include <functional>

namespace vm
{
/**
	 * \brief This class is an adapter for the LVDReader.
	 */

struct PhysicalMemoryBlockIndex	 // DataBlock start in 3d texture
{
	using internal_type = int;
	const internal_type x, y, z;

private:
	uint8_t unit = 0;

public:
	PhysicalMemoryBlockIndex( internal_type x_ = -1,
							  internal_type y_ = -1,
							  internal_type z_ = -1 ) :
	  x( x_ ),
	  y( y_ ),
	  z( z_ ),
	  unit( 0 ) {}
	PhysicalMemoryBlockIndex( internal_type x_,
							  internal_type y_,
							  internal_type z_,
							  uint8_t unit ) :
	  x( x_ ),
	  y( y_ ),
	  z( z_ ),
	  unit( unit ) {}
	int GetPhysicalStorageUnit() const { return unit; }
	void SetPhysicalStorageUnit( uint8_t u ) { unit = u; }
	Vec3i ToVec3i() const { return Vec3i{ x, y, z }; }
};

struct VirtualMemoryBlockIndex
{
	VirtualMemoryBlockIndex() = default;
	VirtualMemoryBlockIndex( std::size_t linearID, int xb, int yb, int zb )
	{
		z = linearID / ( xb * yb );
		y = ( linearID - z * xb * yb ) / xb;
		x = linearID - z * xb * yb - y * xb;
	}
	VirtualMemoryBlockIndex( int x, int y, int z ) :
	  x( x ),
	  y( y ),
	  z( z ) {}
	Vec3i ToVec3i() const { return Vec3i{ x, y, z }; }

	using index_type = int;
	index_type x = -1, y = -1, z = -1;
};
struct SamplePoint{
	float x,y,z;
	SamplePoint(float a,float b,float c):x(a),y(b),z(c){}
};

class Disk3DPageAdapter__pImpl;

class VMFOUNDATION_EXPORTS Disk3DPageAdapter : public AbstrMemoryCache
{
	VM_DECL_IMPL( Disk3DPageAdapter )

public:
	Disk3DPageAdapter( ::vm::IRefCnt *cnt, const std::string &fileName );
	const void *GetPage( size_t pageID ) override;
	size_t GetPageSize() const override;
	size_t GetPhysicalPageCount() const override;
	size_t GetVirtualPageCount() const override;

	int GetPadding() const;
	Size3 GetDataSizeWithoutPadding() const;
	Size3 Get3DPageSize() const;
	int Get3DPageSizeInLog() const;
	Size3 Get3DPageCount() const;
	void *GetRawData() override;
	~Disk3DPageAdapter();

private:
	void *GetPageStorage_Implement( size_t pageID ) override { return nullptr; }
};

class Block3DCache__pImpl;
class VMFOUNDATION_EXPORTS Block3DCache : public AbstrMemoryCache
{
	VM_DECL_IMPL( Block3DCache )

	[[deprecated]] int blockCoordinateToBlockId( int xBlock, int yBlock, int zBlock ) const;
	void Create( I3DBlockDataInterface *pageFile );

public:
	Block3DCache( IRefCnt *cnt, I3DBlockDataInterface *pageFile, std::function<Size3( I3DBlockDataInterface * )> evaluator );
	Block3DCache( IRefCnt *cnt, I3DBlockDataInterface *pageFile );

	void SetDiskFileCache( I3DBlockDataInterface *diskCache );

	Size3 CPUCacheBlockSize() const;
	vm::Size3 CPUCacheSize() const;

	[[deprecated]] int Padding() const;
	[[deprecated]] Size3 DataSizeWithoutPadding() const;
	[[deprecated]] Size3 BlockDim() const;
	[[deprecated]] Size3 BlockSize() const;

	Size3 CacheBlockDim() const;
	size_t GetPhysicalPageCount() const override { return CacheBlockDim().Prod(); }
	size_t GetVirtualPageCount() const override { return BlockDim().Prod(); }
	size_t GetPageSize() const override { return BlockSize().Prod() * sizeof( char ); }

	const void *GetPage( int xBlock, int yBlock, int zBlock ) { return AbstrMemoryCache::GetPage( blockCoordinateToBlockId( xBlock, yBlock, zBlock ) ); }
	const void *GetPage( const VirtualMemoryBlockIndex &index ) { return GetPage( index.x, index.y, index.z ); };

	float SampleBlock(int xBlock,int yBlock,int zBlock,const SamplePoint* sp);
	float SampleBlock(size_t flatID, const SamplePoint* sp);
	float VirtualSample(const SamplePoint* sp);

	void *GetRawData() override;

	virtual ~Block3DCache();

protected:
	[[deprecated]] int GetLog() const;
	void *GetPageStorage_Implement( size_t pageID ) override;
};
}  // namespace vm
