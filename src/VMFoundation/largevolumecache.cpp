
#include <VMFoundation/largevolumecache.h>
#include <VMUtils/log.hpp>
#include <VMFoundation/cachepolicy.h>
#include <VMFoundation/pluginloader.h>
#include <VMUtils/vmnew.hpp>

#include <3rdparty/rapidjson/document.h>
#include <iostream>
#include <cassert>

#define SHOW_LIST_STATE                                                                                          \
	std::cout << "LRU State:" << std::endl;                                                                      \
	for ( auto it = m_lruList.begin(); it != m_lruList.end(); ++it ) {                                           \
		if ( it->hashIter != m_blockIdInCache.end() )                                                            \
			std::cout << "[BlockID:" << it->hashIter->first << " -> CacheID:" << it->blockCacheIndex << "]--->"; \
	}                                                                                                            \
	std::cout << std::endl;

namespace vm
{

class Block3DCache__pImpl
{
	VM_DECL_API( Block3DCache )
public:
	Block3DCache__pImpl( Block3DCache *api ) :
	  q_ptr( api ) {}
	Size3 cacheDim;
	std::unique_ptr<IBlock3DArrayAdapter> m_volumeCache;
	Ref<I3DBlockFilePluginInterface> adapter;
};

class Disk3DPageAdapter__pImpl
{
	VM_DECL_API( Disk3DPageAdapter )
public:
	Disk3DPageAdapter__pImpl( Disk3DPageAdapter *api ,const std::string & fileName):q_ptr( api ),lvdReader( fileName ) {}
	LVDReader lvdReader;
};

Disk3DPageAdapter::Disk3DPageAdapter( ::vm::IRefCnt *cnt, const std::string &fileName ):
	AbstrMemoryCache( cnt ),d_ptr( new Disk3DPageAdapter__pImpl(this,fileName) )
{
}

const void *Disk3DPageAdapter::GetPage( size_t pageID )
{
	VM_IMPL( Disk3DPageAdapter )
	return _->lvdReader.ReadBlock( pageID );
}

size_t Disk3DPageAdapter::GetPageSize() const
{
	const auto _ = d_func();
	return _->lvdReader.BlockSize();
}

size_t Disk3DPageAdapter::GetPhysicalPageCount() const
{

	const auto _ = d_func();
	return _->lvdReader.BlockCount();
}

size_t Disk3DPageAdapter::GetVirtualPageCount() const
{

	const auto _ = d_func();
	return _->lvdReader.BlockCount();
}

int Disk3DPageAdapter::GetPadding() const
{

	const auto _ = d_func();
	return _->lvdReader.GetBlockPadding();
}

Size3 Disk3DPageAdapter::GetDataSizeWithoutPadding() const
{
	const auto _ = d_func();
	return _->lvdReader.OriginalDataSize();
}

Size3 Disk3DPageAdapter::Get3DPageSize() const
{

	const auto _ = d_func();
	const std::size_t len = _->lvdReader.BlockSize();
	return Size3{ len, len, len };
}

int Disk3DPageAdapter::Get3DPageSizeInLog() const
{

	const auto _ = d_func();
	return _->lvdReader.BlockSizeInLog();
}

Size3 Disk3DPageAdapter::Get3DPageCount() const
{

	const auto _ = d_func();
	return _->lvdReader.SizeByBlock();
}

int Block3DCache::blockCoordinateToBlockId( int xBlock, int yBlock, int zBlock ) const
{
	//const auto size = lvdReader.SizeByBlock();
	const auto _ = d_func();
	const auto size = _->adapter->Get3DPageCount();
	const auto x = size.x, y = size.y, z = size.z;
	return zBlock * x * y + yBlock * x + xBlock;
}

void Block3DCache::Create( I3DBlockFilePluginInterface *pageFile )
{
	//const auto p = dynamic_cast<I3DBlockFilePluginInterface *>( GetNextLevelCache() );
	VM_IMPL( Block3DCache )
	const int log = pageFile->Get3DPageSizeInLog();
	//const auto cacheSize = cacheDim*Size3( 1 << log, 1 << log, 1 << log );
	switch ( log ) {
	case 6: _->m_volumeCache = std::make_unique<Int8Block64Cache>( _->cacheDim.x, _->cacheDim.y, _->cacheDim.z, nullptr ); break;
	case 7: _->m_volumeCache = std::make_unique<Int8Block128Cache>( _->cacheDim.x, _->cacheDim.y, _->cacheDim.z, nullptr ); break;
	case 8: _->m_volumeCache = std::make_unique<Int8Block256Cache>( _->cacheDim.x, _->cacheDim.y, _->cacheDim.z, nullptr ); break;
	case 9: _->m_volumeCache = std::make_unique<Int8Block512Cache>( _->cacheDim.x, _->cacheDim.y, _->cacheDim.z, nullptr ); break;
	case 10: _->m_volumeCache = std::make_unique<Int8Block1024Cache>( _->cacheDim.x, _->cacheDim.y, _->cacheDim.z, nullptr ); break;
	default:
		Warning( "Unsupported Cache block Size\n" );
		break;
	}
	if ( !_->m_volumeCache ) {
		std::cerr << "Can not allocate memory for cache\n";
		exit( 0 );
	}
}

Block3DCache::Block3DCache( ::vm::IRefCnt *cnt, const std::string &fileName, std::function<Size3( I3DBlockFilePluginInterface * )> evaluator ) :
  AbstrMemoryCache( cnt ),
  d_ptr( new Block3DCache__pImpl( this ) )
{
	VM_IMPL( Block3DCache )

	const auto cap = fileName.substr( fileName.find_last_of( '.' ) );
	auto p = PluginLoader::CreatePlugin<I3DBlockFilePluginInterface>( cap );
	if ( p == nullptr ) {
		throw std::runtime_error( "Failed to load the plugin that is able to read " + cap + "file" );
	}
	p->Open( fileName );

	_->cacheDim = evaluator( p );

	SetDiskFileCache( p );
	SetCachePolicy( VM_NEW<LRUCachePolicy>() );
}

int Block3DCache::GetLog() const
{
	const auto p = dynamic_cast<const I3DBlockFilePluginInterface *>( GetNextLevelCache() );
	assert( p );
	const int log = p->Get3DPageSizeInLog();
	return log;
}

void *Block3DCache::GetPageStorage_Implement( size_t pageID )
{
	VM_IMPL( Block3DCache )
	return _->m_volumeCache->GetBlockData( pageID );
}

Block3DCache::Block3DCache( ::vm::IRefCnt *cnt, const std::string &fileName ) :
  Block3DCache( cnt, fileName, []( I3DBlockFilePluginInterface *pageFile ) {
	  const auto ps = pageFile->GetPageSize();
	  constexpr size_t maxMemory = 1024 * 1024 * 1024;  // 1GB
	  const auto d = maxMemory / ps;
	  return Size3{ d, 1, 1 };
  } )
{
}

Block3DCache::Block3DCache( ::vm::IRefCnt *cnt, I3DBlockFilePluginInterface *pageFile, std::function<Size3( I3DBlockFilePluginInterface * )> evaluator ) :
  AbstrMemoryCache( cnt )
{
	VM_IMPL( Block3DCache )
	_->cacheDim = evaluator( pageFile );
	SetDiskFileCache( pageFile );
	SetCachePolicy( VM_NEW<LRUCachePolicy>() );
}

Block3DCache::Block3DCache( ::vm::IRefCnt *cnt, I3DBlockFilePluginInterface *pageFile ) :
  Block3DCache( cnt, pageFile, []( I3DBlockFilePluginInterface *pageFile ) {
	  const auto ps = pageFile->GetPageSize();
	  constexpr size_t maxMemory = 1024 * 1024 * 1024;  // 1GB
	  const auto d = maxMemory / ps;
	  return Size3{ d, 1, 1 };
  } )
{
}

void Block3DCache::SetDiskFileCache( I3DBlockFilePluginInterface *diskCache )
{
	VM_IMPL( Block3DCache )
	SetNextLevelCache( diskCache );
	_->adapter = diskCache;
	Create( diskCache );
}

Size3 Block3DCache::CPUCacheBlockSize() const
{
	return Size3( 1 << GetLog(), 1 << GetLog(), 1 << GetLog() );
}

Size3 Block3DCache::CPUCacheSize() const
{
	return CacheBlockDim() * ( 1 << GetLog() );
}

Size3 Block3DCache::BlockSize() const
{
	const auto _ = d_func();
	return _->adapter->Get3DPageSize();
}

Size3 Block3DCache::CacheBlockDim() const
{
	const auto _ = d_func();
	return _->cacheDim;
}

int Block3DCache::Padding() const
{
	const auto _ = d_func();
	return _->adapter->GetPadding();
}

Size3 Block3DCache::DataSizeWithoutPadding() const
{
	const auto _ = d_func();
	return _->adapter->GetDataSizeWithoutPadding();
}

Size3 Block3DCache::BlockDim() const
{
	const auto _ = d_func();
	return _->adapter->Get3DPageCount();
}

}  // namespace vm