
#pragma once

/*
 * This file is temporary.
 */

#include <VMat/geometry.h>
#include <VMFoundation/virtualmemorymanager.h>
#include <VMFoundation/largevolumecache.h>

#include <vector>

namespace vm
{
struct PageDirectoryEntryAbstractIndex
{
	using internal_type = int;
	const internal_type x, y, z;
	PageDirectoryEntryAbstractIndex( internal_type x_ = -1,
									 internal_type y_ = -1,
									 internal_type z_ = -1 ) :
	  x( x_ ),
	  y( y_ ),
	  z( z_ ) {}
};

struct PageTableEntryAbstractIndex
{
	using internal_type = int;
	internal_type x, y, z;
	internal_type lod = 0;
	PageTableEntryAbstractIndex( internal_type x_ = -1,
								 internal_type y_ = -1,
								 internal_type z_ = -1 ) :
	  x( x_ ),
	  y( y_ ),
	  z( z_ ) {}
};




struct BlockDescriptor
{
private:
	PhysicalMemoryBlockIndex value;
	VirtualMemoryBlockIndex key;
public:
	BlockDescriptor() = default;
	BlockDescriptor( const PhysicalMemoryBlockIndex &value, VirtualMemoryBlockIndex key ) :
	  value( value ),
	  key( key ) {}
	const PhysicalMemoryBlockIndex &Value() const { return value; }
	const VirtualMemoryBlockIndex &Key() const { return key; }
};

enum EntryMapFlag
{
	EM_UNKNOWN = 0,
	EM_UNMAPPED = 2,
	EM_MAPPED = 1
};

struct IVideoMemoryParamsEvaluator
{
	virtual Size3 EvalPhysicalTextureSize() const = 0;
	virtual int EvalPhysicalTextureCount() const = 0;
	virtual Size3 EvalPhysicalBlockDim() const = 0;
	virtual ~IVideoMemoryParamsEvaluator() = default;
};

struct LODPageTableInfo
{
	Vec3i virtualSpaceSize;
	void *external = nullptr;
	size_t offset = 0;
};


class MappingTableManager__pImpl;

class VMFOUNDATION_EXPORTS MappingTableManager
{
	VM_DECL_IMPL( MappingTableManager )

public:
	struct PageDirEntry
	{
		int x, y, z, w;
	};
	struct PageTableEntry
	{
		int x, y, z;
	private:
		int w = 0;
	public:
		void SetMapFlag( EntryMapFlag f ) { w = ( w & ( 0xFFF0 ) ) | ( ( 0xF ) & f ); }	// [0,4) bits
		void SetTextureUnit( int unit ) { w = ( w & 0xFF0F ) | ( ( 0xF & unit ) << 4 ); }  // [4,8) bits
		EntryMapFlag GetMapFlag() const { return EntryMapFlag( ( 0xF ) & w ); }
		int GetTextureUnit() const { return ( w >> 4 ) & 0xF; }
	};
	
	using size_type = std::size_t;
	/**
			 * \brief
			 * \param virtualSpaceSize virtual space size
			 */
	MappingTableManager( const std::vector<LODPageTableInfo> &infos, const Size3 &physicalSpaceSize, int physicalSpaceCount );

	const void *GetData( int lod ) const;

	size_t GetBytes( int lod );

	int GetResidentBlocks( int lod );

	~MappingTableManager();

	/**
	* \brief Translates the virtual space address to the physical address and update the mapping table by LRU policy
	*/
	std::vector<PhysicalMemoryBlockIndex> UpdatePageTable( int lod, const std::vector<VirtualMemoryBlockIndex> &missedBlockIndices );
};

struct LVDFileInfo
{
	std::vector<std::string> fileNames;
	float samplingRate = 0.001;
	Vec3f spacing = Vec3f{ 1.f, 1.f, 1.f };
};

struct _std140_layout_LODInfo
{
	Vec4i pageTableSize;
	Vec4i volumeDataSizeNoRepeat;
	Vec4i blockDataSizeNoRepeat;
	uint32_t pageTableOffset;
	uint32_t hashBufferOffset;
	uint32_t idBufferOffset;
	uint32_t pad[ 1 ];
};


}  // namespace ysl
