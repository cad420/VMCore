
#pragma once
#include "foundation_config.h"
#include <VMUtils/concepts.hpp>
#include <VMUtils/common.h>

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

class MemoryAllocationTracker__pImpl;

class VMFOUNDATION_EXPORTS MemoryAllocationTracker : NoCopy
{
	
	VM_DECL_IMPL( MemoryAllocationTracker )

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

	bool IsFull() const;;
	bool IsEmpty() const;;
	size_t GetMaxSize() const;
	size_t GetFreeSize() const;
	size_t GetUsedSize() const;

private:
	void AddNewBlock( size_t Offset, size_t Size );

	void ResetCurrAlignment();


	// When adding new members, do not forget to update move ctor
};

}  // namespace vm
