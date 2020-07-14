
#include <VMFoundation/memoryallocationtracker.h>
#include <VMat/numeric.h>
#include <cassert>
#include <map>

namespace vm
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
                std::less<size_t> // Standard ordering
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



class MemoryAllocationTracker__pImpl
{
	VM_DECL_API( MemoryAllocationTracker )
public:
	MemoryAllocationTracker__pImpl( MemoryAllocationTracker *api ,size_t MaxSize) :
	  q_ptr( api ),m_MaxSize( MaxSize ),m_FreeSize( MaxSize )
	{}
	
	TFreeBlocksByOffsetMap m_FreeBlocksByOffset;
	TFreeBlocksBySizeMap m_FreeBlocksBySize;

	size_t m_MaxSize = 0;
	size_t m_FreeSize = 0;
	size_t m_CurrAlignment = 0;
};

MemoryAllocationTracker::MemoryAllocationTracker( size_t MaxSize ) :
  d_ptr( new MemoryAllocationTracker__pImpl( this, MaxSize) )
{
	VM_IMPL( MemoryAllocationTracker )
	// Insert single maximum-size block
	AddNewBlock( 0, _->m_MaxSize );
	ResetCurrAlignment();
}

MemoryAllocationTracker::~MemoryAllocationTracker()
{
}

MemoryAllocationTracker::MemoryAllocationTracker( MemoryAllocationTracker &&rhs ) noexcept :d_ptr(std::move(const_cast<std::unique_ptr<MemoryAllocationTracker__pImpl>&>(rhs.d_ptr)))
  //m_FreeBlocksByOffset( std::move( rhs.m_FreeBlocksByOffset ) ),
  //m_FreeBlocksBySize( std::move( rhs.m_FreeBlocksBySize ) ),
  //m_MaxSize( rhs.m_MaxSize ),
  //m_FreeSize( rhs.m_FreeSize ),
  //m_CurrAlignment( rhs.m_CurrAlignment )
{
	//rhs.m_MaxSize = 0;
	//rhs.m_FreeSize = 0;
	//rhs.m_CurrAlignment = 0;
	rhs.d_ptr->m_MaxSize = 0;
	rhs.d_ptr->m_FreeSize = 0;
	rhs.d_ptr->m_CurrAlignment = 0;
}

Allocation MemoryAllocationTracker::Allocate( size_t Size, size_t Alignment )
{
	VM_IMPL( MemoryAllocationTracker )
	
	assert( IsPowerOfTwo( Alignment ) );
	Size = Align( Size, Alignment );
	if ( _->m_FreeSize < Size )
		return Allocation::InvalidAllocation();

	auto AlignmentReserve = ( Alignment > _->m_CurrAlignment ) ? Alignment - _->m_CurrAlignment : 0;
	// Get the first block that is large enough to encompass Size + AlignmentReserve bytes
	// lower_bound() returns an iterator pointing to the first element that
	// is not less (i.e. >= ) than key
	auto SmallestBlockItIt = _->m_FreeBlocksBySize.lower_bound( Size + AlignmentReserve );
	if ( SmallestBlockItIt == _->m_FreeBlocksBySize.end() )
		return Allocation::InvalidAllocation();

	auto SmallestBlockIt = SmallestBlockItIt->second;
	//VERIFY_EXPR(Size + AlignmentReserve <= SmallestBlockIt->second.Size);
	//VERIFY_EXPR(SmallestBlockIt->second.Size == SmallestBlockItIt->first);

	//     SmallestBlockIt.Offset
	//        |                                  |
	//        |<------SmallestBlockIt.Size------>|
	//        |<------Size------>|<---NewSize--->|
	//        |                  |
	//      Offset              NewOffset
	//
	auto Offset = SmallestBlockIt->first;
	//VERIFY_EXPR(Offset % m_CurrAlignment == 0);
	auto AlignedOffset = Align( Offset, Alignment );
	auto AdjustedSize = Size + ( AlignedOffset - Offset );
	//VERIFY_EXPR(AdjustedSize <= Size + AlignmentReserve);
	auto NewOffset = Offset + AdjustedSize;
	auto NewSize = SmallestBlockIt->second.Size - AdjustedSize;
	//VERIFY_EXPR(SmallestBlockItIt == SmallestBlockIt->second.OrderBySizeIt);
	_->m_FreeBlocksBySize.erase( SmallestBlockItIt );
	_->m_FreeBlocksByOffset.erase( SmallestBlockIt );
	if ( NewSize > 0 ) {
		AddNewBlock( NewOffset, NewSize );
	}

	_->m_FreeSize -= AdjustedSize;

	if ( ( Size & ( _->m_CurrAlignment - 1 ) ) != 0 ) {
		if ( vm::IsPowerOfTwo( Size ) ) {
			//VERIFY_EXPR(Size >= Alignment && Size < m_CurrAlignment);
			_->m_CurrAlignment = Size;
		} else {
			_->m_CurrAlignment = std::min( _->m_CurrAlignment, Alignment );
		}
	}
	return Allocation{ Offset, AdjustedSize };
}

void MemoryAllocationTracker::Free( size_t Offset, size_t Size )
{
	VM_IMPL( MemoryAllocationTracker )
	
	//VERIFY_EXPR(Offset + Size <= m_MaxSize);

	// Find the first element whose offset is greater than the specified offset.
	// upper_bound() returns an iterator pointing to the first element in the
	// container whose key is considered to go after k.
	auto NextBlockIt = _->m_FreeBlocksByOffset.upper_bound( Offset );
	// Block being deallocated must not overlap with the next block
	//VERIFY_EXPR(NextBlockIt == m_FreeBlocksByOffset.end() || Offset + Size <= NextBlockIt->first);
	auto PrevBlockIt = NextBlockIt;
	if ( PrevBlockIt != _->m_FreeBlocksByOffset.begin() ) {
		--PrevBlockIt;
		// Block being deallocated must not overlap with the previous block
		//VERIFY_EXPR(Offset >= PrevBlockIt->first + PrevBlockIt->second.Size);
	} else
		PrevBlockIt = _->m_FreeBlocksByOffset.end();

	size_t NewSize, NewOffset;
	if ( PrevBlockIt != _->m_FreeBlocksByOffset.end() && Offset == PrevBlockIt->first + PrevBlockIt->second.Size ) {
		//  PrevBlock.Offset             Offset
		//       |                          |
		//       |<-----PrevBlock.Size----->|<------Size-------->|
		//
		NewSize = PrevBlockIt->second.Size + Size;
		NewOffset = PrevBlockIt->first;

		if ( NextBlockIt != _->m_FreeBlocksByOffset.end() && Offset + Size == NextBlockIt->first ) {
			//   PrevBlock.Offset           Offset            NextBlock.Offset
			//     |                          |                    |
			//     |<-----PrevBlock.Size----->|<------Size-------->|<-----NextBlock.Size----->|
			//
			NewSize += NextBlockIt->second.Size;
			_->m_FreeBlocksBySize.erase( PrevBlockIt->second.OrderBySizeIt );
			_->m_FreeBlocksBySize.erase( NextBlockIt->second.OrderBySizeIt );
			// Delete the range of two blocks
			++NextBlockIt;
			_->m_FreeBlocksByOffset.erase( PrevBlockIt, NextBlockIt );
		} else {
			//   PrevBlock.Offset           Offset                     NextBlock.Offset
			//     |                          |                             |
			//     |<-----PrevBlock.Size----->|<------Size-------->| ~ ~ ~  |<-----NextBlock.Size----->|
			//
			_->m_FreeBlocksBySize.erase( PrevBlockIt->second.OrderBySizeIt );
			_->m_FreeBlocksByOffset.erase( PrevBlockIt );
		}
	} else if ( NextBlockIt != _->m_FreeBlocksByOffset.end() && Offset + Size == NextBlockIt->first ) {
		//   PrevBlock.Offset                   Offset            NextBlock.Offset
		//     |                                  |                    |
		//     |<-----PrevBlock.Size----->| ~ ~ ~ |<------Size-------->|<-----NextBlock.Size----->|
		//
		NewSize = Size + NextBlockIt->second.Size;
		NewOffset = Offset;
		_->m_FreeBlocksBySize.erase( NextBlockIt->second.OrderBySizeIt );
		_->m_FreeBlocksByOffset.erase( NextBlockIt );
	} else {
		//   PrevBlock.Offset                   Offset                     NextBlock.Offset
		//     |                                  |                            |
		//     |<-----PrevBlock.Size----->| ~ ~ ~ |<------Size-------->| ~ ~ ~ |<-----NextBlock.Size----->|
		//
		NewSize = Size;
		NewOffset = Offset;
	}

	AddNewBlock( NewOffset, NewSize );

	_->m_FreeSize += Size;
	if ( IsEmpty() ) {
		// Reset current alignment
		//VERIFY_EXPR(DbgGetNumFreeBlocks() == 1);
		ResetCurrAlignment();
	}
}

bool MemoryAllocationTracker::IsFull() const
{
	const auto _ = d_func();
	return _->m_FreeSize == 0;
}

bool MemoryAllocationTracker::IsEmpty() const
{
	const auto _ = d_func();
	return _->m_FreeSize == _->m_MaxSize;
}

size_t MemoryAllocationTracker::GetMaxSize() const
{
	const auto _ = d_func();
	return _->m_MaxSize;
}

size_t MemoryAllocationTracker::GetFreeSize() const
{
	const auto _ = d_func();
	return _->m_FreeSize;
}


size_t MemoryAllocationTracker::GetUsedSize() const
{
	const auto _ = d_func();
	return _->m_MaxSize - _->m_FreeSize;
}

void MemoryAllocationTracker::AddNewBlock( size_t Offset, size_t Size )
{
	VM_IMPL( MemoryAllocationTracker )
	auto NewBlockIt = _->m_FreeBlocksByOffset.emplace( Offset, Size );
	//VERIFY_EXPR(NewBlockIt.second);
	auto OrderIt = _->m_FreeBlocksBySize.emplace( Size, NewBlockIt.first );
	NewBlockIt.first->second.OrderBySizeIt = OrderIt;
}

void MemoryAllocationTracker::ResetCurrAlignment()
{
	VM_IMPL( MemoryAllocationTracker )
	for ( _->m_CurrAlignment = 1; _->m_CurrAlignment * 2 <= _->m_MaxSize; _->m_CurrAlignment *= 2 )
		;
}
}  // namespace ysl
