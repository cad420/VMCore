
#include <VMFoundation/mappingtablemanager.h>
#include <VMFoundation/lineararray.h>
#include <unordered_map>

namespace vm
{

class MappingTableManager__pImpl
{
	VM_DECL_API( MappingTableManager )
public:
	using LRUList = std::list<std::pair<PageTableEntryAbstractIndex, PhysicalMemoryBlockIndex>>;
	using LRUMap = std::unordered_map<size_t, LRUList::iterator>;

	MappingTableManager__pImpl( MappingTableManager *api ):
	  q_ptr( api ) {}
	std::vector<Linear3DArray<MappingTableManager::PageTableEntry>> lodPageTables;
	LRUMap lruMap;
	LRUList lruList;
	std::vector<size_t> blocks;
};

MappingTableManager::MappingTableManager( const std::vector<LODPageTableInfo> &infos, const Size3 &physicalSpaceSize, int physicalSpaceCount ):
d_ptr( new MappingTableManager__pImpl(this))
{
	VM_IMPL( MappingTableManager )
	
	const int lod = infos.size();
	_->lodPageTables.resize( lod );
	_->blocks.resize( lod );

	// lod page table
	for ( int i = 0; i < lod; i++ ) {
		if ( infos[ i ].external == nullptr ) _->lodPageTables[ i ] = Linear3DArray<PageTableEntry>( Size3( infos[ i ].virtualSpaceSize ), nullptr );
		else _->lodPageTables[ i ] = Linear3DArray<PageTableEntry>( infos[ i ].virtualSpaceSize.x, infos[ i ].virtualSpaceSize.y, infos[ i ].virtualSpaceSize.z, (PageTableEntry *)infos[ i ].external, false );
		size_t blockId = 0;

		for ( auto z = 0; z < _->lodPageTables[ i ].Size().z; z++ )
			for ( auto y = 0; y < _->lodPageTables[ i ].Size().y; y++ )
				for ( auto x = 0; x < _->lodPageTables[ i ].Size().x; x++ ) {
					PageTableEntry entry;
					entry.x = -1;
					entry.y = -1;
					entry.z = -1;
					entry.SetMapFlag( EM_UNMAPPED );
					//entry.w = EM_UNMAPPED;
					( _->lodPageTables[ i ] )( x, y, z ) = entry;

					_->lruMap[ blockId++ ] = _->lruList.end();
				}
	}

	// lod lru list

	for ( int i = 0; i < physicalSpaceCount; i++ )
		for ( auto z = 0; z < physicalSpaceSize.z; z++ )
			for ( auto y = 0; y < physicalSpaceSize.y; y++ )
				for ( auto x = 0; x < physicalSpaceSize.x; x++ ) {
					_->lruList.emplace_back(
						PageTableEntryAbstractIndex( -1, -1, -1 ),
						PhysicalMemoryBlockIndex( x, y, z, i ) );
				}
}

const void *MappingTableManager::GetData( int lod ) const
{
	const auto _ = d_func();
	assert( lod < _->lodPageTables.size() );
	return _->lodPageTables[ lod ].Data();
}

size_t MappingTableManager::GetBytes( int lod )
{
	VM_IMPL( MappingTableManager )
	return _->lodPageTables[ lod ].Size().Prod() * sizeof( PageTableEntry );
}

int MappingTableManager::GetResidentBlocks( int lod )
{
	VM_IMPL( MappingTableManager )
	return _->blocks[ lod ];
}

MappingTableManager::~MappingTableManager()
{
}

std::vector<PhysicalMemoryBlockIndex> MappingTableManager::UpdatePageTable( int lod, const std::vector<VirtualMemoryBlockIndex> &missedBlockIndices )
{
	VM_IMPL( MappingTableManager )
	
	const auto missedBlocks = missedBlockIndices.size();
	std::vector<PhysicalMemoryBlockIndex> physicalIndices;
	physicalIndices.reserve( missedBlocks );
	// Update LRU List
	for ( int i = 0; i < missedBlocks; i++ ) {
		const auto &index = missedBlockIndices[ i ];
		auto &pageTableEntry = _->lodPageTables[ lod ]( index.x, index.y, index.z );
		const size_t flatBlockID = index.z * _->lodPageTables[ lod ].Size().x * _->lodPageTables[ lod ].Size().y + index.y * _->lodPageTables[ lod ].Size().x + index.x;
		if ( pageTableEntry.GetMapFlag() == EM_MAPPED ) {
			// move the already mapped node to the head
			_->lruList.splice( _->lruList.begin(), _->lruList, _->lruMap[ flatBlockID ] );

		} else {
			auto &last = _->lruList.back();
			//pageTableEntry.w = EntryMapFlag::EM_MAPPED; // Map the flag of page table entry
			pageTableEntry.SetMapFlag( EM_MAPPED );
			// last.second is the cache block index
			physicalIndices.push_back( last.second );
			pageTableEntry.x = last.second.x; // fill the page table entry
			pageTableEntry.y = last.second.y;
			pageTableEntry.z = last.second.z;
			pageTableEntry.SetTextureUnit( last.second.GetPhysicalStorageUnit() );
			if ( last.first.x != -1 ) // detach previous mapped storage
			{
				_->lodPageTables[ last.first.lod ]( last.first.x, last.first.y, last.first.z ).SetMapFlag( EM_UNMAPPED );
				_->lruMap[ flatBlockID ] = _->lruList.end();
				_->blocks[ last.first.lod ]--;
			}
			// critical section : last
			last.first.x = index.x;
			last.first.y = index.y;
			last.first.z = index.z;

			last.first.lod = lod; //

			_->lruList.splice( _->lruList.begin(), _->lruList, --_->lruList.end() ); // move from tail to head, LRU policy
			_->lruMap[ flatBlockID ] = _->lruList.begin();
			_->blocks[ lod ]++;
		}
	}
	return physicalIndices;
}
}
