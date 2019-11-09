
#pragma once

#include <VMat/geometry.h>
#include "virtualmemorymanager.h"

namespace ysl
{

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
	//	virtual int EvalHashBufferSize()const = 0;
	//	virtual int EvalIDBufferCount()const = 0;
	virtual ~IVideoMemoryParamsEvaluator() = default;
};

struct DefaultMemoryParamsEvaluator : IVideoMemoryParamsEvaluator
{
private:
	const Size3 virtualDim;
	const Size3 blockSize;
	const std::size_t videoMem;
	int textureUnitCount = 0;
	Size3 finalBlockDim = { 0, 0, 0 };

public:
	DefaultMemoryParamsEvaluator( const ysl::Size3 &virtualDim, const Size3 &blockSize, std::size_t videoMemory ):virtualDim( virtualDim ),blockSize( blockSize ),videoMem( videoMemory )
	{
		std::size_t d = 0;
		textureUnitCount = 1;
		while ( ++d ) {
			const auto memory = d * d * d * blockSize.Prod();
			if ( memory >= videoMem * 1024 )
				break;
		}
		while ( d > 10 ) {
			d /= 2;
			textureUnitCount++;
		}
		finalBlockDim = Size3{ d, d, d };
	}

	Size3 EvalPhysicalTextureSize() const override
	{
		return blockSize * EvalPhysicalBlockDim();
	}
	Size3 EvalPhysicalBlockDim() const override
	{
		return { 10, 10, 10 };
	}

	int EvalPhysicalTextureCount() const override
	{
		return 3;
	}
	~DefaultMemoryParamsEvaluator() = default;
};

struct LODPageTableInfo
{
	Vec3i virtualSpaceSize;
	void *external = nullptr;
	size_t offset = 0;
};

class MappingTableManager
{
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

private:
	using LRUList = std::list<std::pair<PageTableEntryAbstractIndex, PhysicalMemoryBlockIndex>>;
	using LRUMap = std::unordered_map<size_t, LRUList::iterator>;

	Linear3DArray<PageTableEntry> pageTable;

	std::vector<Linear3DArray<PageTableEntry>> lodPageTables;

	LRUMap lruMap;
	LRUList lruList;

	std::vector<size_t> blocks;

	void InitCPUPageTable( const Size3 &blockDim, void *external )
	{
		// Only initialization flag filed, the table entry is determined by cache miss at run time using lazy evaluation policy.
		if ( external == nullptr )
			pageTable = Linear3DArray<PageTableEntry>( blockDim, nullptr );
		else
			pageTable = Linear3DArray<PageTableEntry>( blockDim.x, blockDim.y, blockDim.z, (PageTableEntry *)external, false );
		size_t blockId = 0;
		for ( auto z = 0; z < pageTable.Size().z; z++ )
			for ( auto y = 0; y < pageTable.Size().y; y++ )
				for ( auto x = 0; x < pageTable.Size().x; x++ ) {
					PageTableEntry entry;
					entry.x = -1;
					entry.y = -1;
					entry.z = -1;
					entry.SetMapFlag( EM_UNMAPPED );
					//entry.w = EM_UNMAPPED;
					( pageTable )( x, y, z ) = entry;
					lruMap[ blockId++ ] = lruList.end();
				}
	}
	void InitLRUList( const Size3 &physicalMemoryBlock, int unitCount )
	{
		for ( int i = 0; i < unitCount; i++ )
			for ( auto z = 0; z < physicalMemoryBlock.z; z++ )
				for ( auto y = 0; y < physicalMemoryBlock.y; y++ )
					for ( auto x = 0; x < physicalMemoryBlock.x; x++ ) {
						lruList.emplace_back(
						  PageTableEntryAbstractIndex( -1, -1, -1 ),
						  PhysicalMemoryBlockIndex( x, y, z, i ) );
					}
	}

	//void InitCPUPageTable();

public:
	using size_type = std::size_t;
	/**
			 * \brief
			 * \param virtualSpaceSize virtual space size
			 */
	MappingTableManager( const Size3 &virtualSpaceSize, const Size3 &physicalSpaceSize )
	{
		InitCPUPageTable( virtualSpaceSize, nullptr );
		InitLRUList( physicalSpaceSize, 1 );
	}

	MappingTableManager( const Size3 &virtualSpaceSize, const Size3 &physicalSpaceSize, int physicalSpaceCount )
	{
		InitCPUPageTable( virtualSpaceSize, nullptr );
		InitLRUList( physicalSpaceSize, physicalSpaceCount );
	}

	MappingTableManager( const Size3 &virtualSpaceSize, const Size3 &physicalSpaceSize, int physicalSpaceCount, void *external )
	{
		assert( external );
		InitCPUPageTable( virtualSpaceSize, external );
		InitLRUList( physicalSpaceSize, physicalSpaceCount );
	}

	MappingTableManager( const std::vector<LODPageTableInfo> &infos, const Size3 &physicalSpaceSize, int physicalSpaceCount )
	{
		const int lod = infos.size();
		lodPageTables.resize( lod );
		blocks.resize( lod );

		// lod page table
		for ( int i = 0; i < lod; i++ ) {
			if ( infos[ i ].external == nullptr )
				lodPageTables[ i ] = Linear3DArray<PageTableEntry>( Size3( infos[ i ].virtualSpaceSize ), nullptr );
			else
				lodPageTables[ i ] = Linear3DArray<PageTableEntry>( infos[ i ].virtualSpaceSize.x, infos[ i ].virtualSpaceSize.y, infos[ i ].virtualSpaceSize.z, (PageTableEntry *)infos[ i ].external, false );
			size_t blockId = 0;

			for ( auto z = 0; z < lodPageTables[ i ].Size().z; z++ )
				for ( auto y = 0; y < lodPageTables[ i ].Size().y; y++ )
					for ( auto x = 0; x < lodPageTables[ i ].Size().x; x++ ) {
						PageTableEntry entry;
						entry.x = -1;
						entry.y = -1;
						entry.z = -1;
						entry.SetMapFlag( EM_UNMAPPED );
						//entry.w = EM_UNMAPPED;
						( lodPageTables[ i ] )( x, y, z ) = entry;

						lruMap[ blockId++ ] = lruList.end();
					}
		}

		// lod lru list

		for ( int i = 0; i < physicalSpaceCount; i++ )
			for ( auto z = 0; z < physicalSpaceSize.z; z++ )
				for ( auto y = 0; y < physicalSpaceSize.y; y++ )
					for ( auto x = 0; x < physicalSpaceSize.x; x++ ) {
						lruList.emplace_back(
						  PageTableEntryAbstractIndex( -1, -1, -1 ),
						  PhysicalMemoryBlockIndex( x, y, z, i ) );
					}
	}

	const void *GetData() const { return pageTable.Data(); }

	size_t GetBytes( int lod ) { return lodPageTables[ lod ].Size().Prod() * sizeof( PageTableEntry ); }

	int GetResidentBlocks( int lod ) { return blocks[ lod ]; }

	/**
			 * \brief Translates the virtual space address to the physical address and update the mapping table by LRU policy
			 */
	std::vector<PhysicalMemoryBlockIndex> UpdatePageTable( int lod, const std::vector<VirtualMemoryBlockIndex> &missedBlockIndices )
	{
		const auto missedBlocks = missedBlockIndices.size();
		std::vector<PhysicalMemoryBlockIndex> physicalIndices;
		physicalIndices.reserve( missedBlocks );
		// Update LRU List
		for ( int i = 0; i < missedBlocks; i++ ) {
			const auto &index = missedBlockIndices[ i ];
			auto &pageTableEntry = lodPageTables[ lod ]( index.x, index.y, index.z );
			const size_t flatBlockID = index.z * lodPageTables[ lod ].Size().x * lodPageTables[ lod ].Size().y + index.y * lodPageTables[ lod ].Size().x + index.x;
			if ( pageTableEntry.GetMapFlag() == EM_MAPPED ) {
				// move the already mapped node to the head
				lruList.splice( lruList.begin(), lruList, lruMap[ flatBlockID ] );

			} else {
				auto &last = lruList.back();
				//pageTableEntry.w = EntryMapFlag::EM_MAPPED; // Map the flag of page table entry
				pageTableEntry.SetMapFlag( EM_MAPPED );
				// last.second is the cache block index
				physicalIndices.push_back( last.second );
				pageTableEntry.x = last.second.x;  // fill the page table entry
				pageTableEntry.y = last.second.y;
				pageTableEntry.z = last.second.z;
				pageTableEntry.SetTextureUnit( last.second.GetPhysicalStorageUnit() );
				if ( last.first.x != -1 )  // detach previous mapped storage
				{
					lodPageTables[ last.first.lod ]( last.first.x, last.first.y, last.first.z ).SetMapFlag( EM_UNMAPPED );
					lruMap[ flatBlockID ] = lruList.end();
					blocks[ last.first.lod ]--;
				}
				// critical section : last
				last.first.x = index.x;
				last.first.y = index.y;
				last.first.z = index.z;

				last.first.lod = lod;  //

				lruList.splice( lruList.begin(), lruList, --lruList.end() );  // move from tail to head, LRU policy
				lruMap[ flatBlockID ] = lruList.begin();
				blocks[ lod ]++;
			}
		}
		return physicalIndices;
	}
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
}
