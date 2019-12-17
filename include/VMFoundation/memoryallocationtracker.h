
#pragma once
#include <map>
#include "foundation_config.h"

namespace vm
{
class VMFOUNDATION_EXPORTS MemoryAllocationTracker
{
public:
	using OffsetType = size_t;

private:
	struct FreeBlockInfo;

	// Type of the map that keeps memory blocks sorted by their offsets
	using TFreeBlocksByOffsetMap =
	  std::map<OffsetType,
			   FreeBlockInfo,
			   std::less<OffsetType>  // Standard ordering
			   // Raw memory allocator
			   >;

	// Type of the map that keeps memory blocks sorted by their sizes
	using TFreeBlocksBySizeMap =
	  std::multimap<OffsetType,
					TFreeBlocksByOffsetMap::iterator,
					std::less<OffsetType>  // Standard ordering
					>;

	struct FreeBlockInfo
	{
		// Block size (no reserved space for the size of the allocation)
		OffsetType Size;

		// Iterator referencing this block in the multimap sorted by the block size
		TFreeBlocksBySizeMap::iterator OrderBySizeIt;

		FreeBlockInfo( OffsetType _Size ) :
		  Size( _Size ) {}
	};



public:
	MemoryAllocationTracker( OffsetType MaxSize );
	~MemoryAllocationTracker();
	MemoryAllocationTracker( MemoryAllocationTracker &&rhs ) noexcept;
	MemoryAllocationTracker &operator=( MemoryAllocationTracker &&rhs ) = default;
	MemoryAllocationTracker( const MemoryAllocationTracker & ) = delete;
	MemoryAllocationTracker &operator=( const MemoryAllocationTracker & ) = delete;

	// Offset returned by Allocate() may not be aligned, but the size of the allocation
	// is sufficient to properly align it
	struct Allocation
	{
		Allocation( OffsetType offset, OffsetType size ) :
		  UnalignedOffset( offset ),
		  Size( size )
		{
		}

		Allocation() {}

		static constexpr OffsetType InvalidOffset = static_cast<OffsetType>( -1 );
		static Allocation InvalidAllocation()
		{
			return Allocation{ InvalidOffset, 0 };
		}

		bool IsValid() const
		{
			return UnalignedOffset != InvalidAllocation().UnalignedOffset;
		}

		OffsetType UnalignedOffset = InvalidOffset;
		OffsetType Size = 0;
	};

	Allocation Allocate( OffsetType Size, OffsetType Alignment );

	void Free( Allocation &&allocation )
	{
		Free( allocation.UnalignedOffset, allocation.Size );
		allocation = Allocation{};
	}

	void Free( OffsetType Offset, OffsetType Size );

	bool IsFull() const { return m_FreeSize == 0; };
	bool IsEmpty() const { return m_FreeSize == m_MaxSize; };
	OffsetType GetMaxSize() const { return m_MaxSize; }
	OffsetType GetFreeSize() const { return m_FreeSize; }
	OffsetType GetUsedSize() const { return m_MaxSize - m_FreeSize; }

private:
	void AddNewBlock( OffsetType Offset, OffsetType Size );

	void ResetCurrAlignment();

	TFreeBlocksByOffsetMap m_FreeBlocksByOffset;
	TFreeBlocksBySizeMap m_FreeBlocksBySize;

	OffsetType m_MaxSize = 0;
	OffsetType m_FreeSize = 0;
	OffsetType m_CurrAlignment = 0;
	// When adding new members, do not forget to update move ctor
};
}  // namespace ysl
