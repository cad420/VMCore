
#include <VMFoundation/memoryallocationtracker.h>
#include <VMat/numeric.h>
#include <cassert>

namespace ysl
{

MemoryAllocationTracker::MemoryAllocationTracker( OffsetType MaxSize ) :
  m_MaxSize( MaxSize ),
  m_FreeSize( MaxSize )
{
	// Insert single maximum-size block
	AddNewBlock( 0, m_MaxSize );
	ResetCurrAlignment();
}

MemoryAllocationTracker::~MemoryAllocationTracker()
{
}

MemoryAllocationTracker::MemoryAllocationTracker( MemoryAllocationTracker &&rhs ) noexcept :
  m_FreeBlocksByOffset( std::move( rhs.m_FreeBlocksByOffset ) ),
  m_FreeBlocksBySize( std::move( rhs.m_FreeBlocksBySize ) ),
  m_MaxSize( rhs.m_MaxSize ),
  m_FreeSize( rhs.m_FreeSize ),
  m_CurrAlignment( rhs.m_CurrAlignment )
{
	rhs.m_MaxSize = 0;
	rhs.m_FreeSize = 0;
	rhs.m_CurrAlignment = 0;
}

MemoryAllocationTracker::Allocation MemoryAllocationTracker::Allocate( OffsetType Size, OffsetType Alignment )
{
	assert( IsPowerOfTwo( Alignment ) );
	Size = Align( Size, Alignment );
	if ( m_FreeSize < Size )
		return Allocation::InvalidAllocation();

	auto AlignmentReserve = ( Alignment > m_CurrAlignment ) ? Alignment - m_CurrAlignment : 0;
	// Get the first block that is large enough to encompass Size + AlignmentReserve bytes
	// lower_bound() returns an iterator pointing to the first element that
	// is not less (i.e. >= ) than key
	auto SmallestBlockItIt = m_FreeBlocksBySize.lower_bound( Size + AlignmentReserve );
	if ( SmallestBlockItIt == m_FreeBlocksBySize.end() )
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
	m_FreeBlocksBySize.erase( SmallestBlockItIt );
	m_FreeBlocksByOffset.erase( SmallestBlockIt );
	if ( NewSize > 0 ) {
		AddNewBlock( NewOffset, NewSize );
	}

	m_FreeSize -= AdjustedSize;

	if ( ( Size & ( m_CurrAlignment - 1 ) ) != 0 ) {
		if ( ysl::IsPowerOfTwo( Size ) ) {
			//VERIFY_EXPR(Size >= Alignment && Size < m_CurrAlignment);
			m_CurrAlignment = Size;
		} else {
			m_CurrAlignment = std::min( m_CurrAlignment, Alignment );
		}
	}
	return Allocation{ Offset, AdjustedSize };
}

void MemoryAllocationTracker::Free( OffsetType Offset, OffsetType Size )
{
	//VERIFY_EXPR(Offset + Size <= m_MaxSize);

	// Find the first element whose offset is greater than the specified offset.
	// upper_bound() returns an iterator pointing to the first element in the
	// container whose key is considered to go after k.
	auto NextBlockIt = m_FreeBlocksByOffset.upper_bound( Offset );
	// Block being deallocated must not overlap with the next block
	//VERIFY_EXPR(NextBlockIt == m_FreeBlocksByOffset.end() || Offset + Size <= NextBlockIt->first);
	auto PrevBlockIt = NextBlockIt;
	if ( PrevBlockIt != m_FreeBlocksByOffset.begin() ) {
		--PrevBlockIt;
		// Block being deallocated must not overlap with the previous block
		//VERIFY_EXPR(Offset >= PrevBlockIt->first + PrevBlockIt->second.Size);
	} else
		PrevBlockIt = m_FreeBlocksByOffset.end();

	OffsetType NewSize, NewOffset;
	if ( PrevBlockIt != m_FreeBlocksByOffset.end() && Offset == PrevBlockIt->first + PrevBlockIt->second.Size ) {
		//  PrevBlock.Offset             Offset
		//       |                          |
		//       |<-----PrevBlock.Size----->|<------Size-------->|
		//
		NewSize = PrevBlockIt->second.Size + Size;
		NewOffset = PrevBlockIt->first;

		if ( NextBlockIt != m_FreeBlocksByOffset.end() && Offset + Size == NextBlockIt->first ) {
			//   PrevBlock.Offset           Offset            NextBlock.Offset
			//     |                          |                    |
			//     |<-----PrevBlock.Size----->|<------Size-------->|<-----NextBlock.Size----->|
			//
			NewSize += NextBlockIt->second.Size;
			m_FreeBlocksBySize.erase( PrevBlockIt->second.OrderBySizeIt );
			m_FreeBlocksBySize.erase( NextBlockIt->second.OrderBySizeIt );
			// Delete the range of two blocks
			++NextBlockIt;
			m_FreeBlocksByOffset.erase( PrevBlockIt, NextBlockIt );
		} else {
			//   PrevBlock.Offset           Offset                     NextBlock.Offset
			//     |                          |                             |
			//     |<-----PrevBlock.Size----->|<------Size-------->| ~ ~ ~  |<-----NextBlock.Size----->|
			//
			m_FreeBlocksBySize.erase( PrevBlockIt->second.OrderBySizeIt );
			m_FreeBlocksByOffset.erase( PrevBlockIt );
		}
	} else if ( NextBlockIt != m_FreeBlocksByOffset.end() && Offset + Size == NextBlockIt->first ) {
		//   PrevBlock.Offset                   Offset            NextBlock.Offset
		//     |                                  |                    |
		//     |<-----PrevBlock.Size----->| ~ ~ ~ |<------Size-------->|<-----NextBlock.Size----->|
		//
		NewSize = Size + NextBlockIt->second.Size;
		NewOffset = Offset;
		m_FreeBlocksBySize.erase( NextBlockIt->second.OrderBySizeIt );
		m_FreeBlocksByOffset.erase( NextBlockIt );
	} else {
		//   PrevBlock.Offset                   Offset                     NextBlock.Offset
		//     |                                  |                            |
		//     |<-----PrevBlock.Size----->| ~ ~ ~ |<------Size-------->| ~ ~ ~ |<-----NextBlock.Size----->|
		//
		NewSize = Size;
		NewOffset = Offset;
	}

	AddNewBlock( NewOffset, NewSize );

	m_FreeSize += Size;
	if ( IsEmpty() ) {
		// Reset current alignment
		//VERIFY_EXPR(DbgGetNumFreeBlocks() == 1);
		ResetCurrAlignment();
	}
}

void MemoryAllocationTracker::AddNewBlock( OffsetType Offset, OffsetType Size )
{
	auto NewBlockIt = m_FreeBlocksByOffset.emplace( Offset, Size );
	//VERIFY_EXPR(NewBlockIt.second);
	auto OrderIt = m_FreeBlocksBySize.emplace( Size, NewBlockIt.first );
	NewBlockIt.first->second.OrderBySizeIt = OrderIt;
}

void MemoryAllocationTracker::ResetCurrAlignment()
{
	for ( m_CurrAlignment = 1; m_CurrAlignment * 2 <= m_MaxSize; m_CurrAlignment *= 2 )
		;
}
}  // namespace ysl
