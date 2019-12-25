
#pragma once
#include <map>
#include "foundation_config.h"
#include <VMUtils/concepts.hpp>

namespace vm
{

struct Allocation
{
	Allocation( size_t offset, size_t size ) :
	  UnalignedOffset( offset ),
	  Size( size )
	{
	}

	Allocation() {}

	static constexpr size_t InvalidOffset = static_cast<size_t>( -1 );
	static Allocation InvalidAllocation()
	{
		return Allocation{ InvalidOffset, 0 };
	}

	bool IsValid() const
	{
		return UnalignedOffset != InvalidAllocation().UnalignedOffset;
	}

	size_t UnalignedOffset = InvalidOffset;
	size_t Size = 0;
};


class VMFOUNDATION_EXPORTS MemoryAllocationTracker:NoCopy
{
	struct FreeBlockInfo;

	// Type of the map that keeps memory blocks sorted by their offsets
	using TFreeBlocksByOffsetMap =
	  std::map<size_t,
			   FreeBlockInfo,
			   std::less<size_t>  // Standard ordering
			   // Raw memory allocator
			   >;

	// Type of the map that keeps memory blocks sorted by their sizes
	using TFreeBlocksBySizeMap =
	  std::multimap<size_t,
					TFreeBlocksByOffsetMap::iterator,
					std::less<size_t>  // Standard ordering
					>;

	struct FreeBlockInfo
	{
		// Block size (no reserved space for the size of the allocation)
		size_t Size;

		// Iterator referencing this block in the multimap sorted by the block size
		TFreeBlocksBySizeMap::iterator OrderBySizeIt;

		FreeBlockInfo( size_t _Size ) :
		  Size( _Size ) {}
	};


public:
	MemoryAllocationTracker( size_t MaxSize );
	~MemoryAllocationTracker();
	MemoryAllocationTracker( MemoryAllocationTracker &&rhs ) noexcept;
	MemoryAllocationTracker &operator=( MemoryAllocationTracker &&rhs ) = default;

	// Offset returned by Allocate() may not be aligned, but the size of the allocation
	// is sufficient to properly align it


	Allocation Allocate( size_t Size, size_t Alignment );

	void Free( Allocation &&allocation )
	{
		Free( allocation.UnalignedOffset, allocation.Size );
		allocation = Allocation{};
	}

	void Free( size_t Offset, size_t Size );

	bool IsFull() const { return m_FreeSize == 0; };
	bool IsEmpty() const { return m_FreeSize == m_MaxSize; };
	size_t GetMaxSize() const { return m_MaxSize; }
	size_t GetFreeSize() const { return m_FreeSize; }
	size_t GetUsedSize() const { return m_MaxSize - m_FreeSize; }

private:
	void AddNewBlock( size_t Offset, size_t Size );

	void ResetCurrAlignment();

	TFreeBlocksByOffsetMap m_FreeBlocksByOffset;
	TFreeBlocksBySizeMap m_FreeBlocksBySize;

	size_t m_MaxSize = 0;
	size_t m_FreeSize = 0;
	size_t m_CurrAlignment = 0;
	// When adding new members, do not forget to update move ctor
};
}  // namespace ysl
