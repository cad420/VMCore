#include <VMFoundation/blockedgridvolumefile.h>
#include <VMat/numeric.h>
#include <VMFoundation/rawstream.h>
#include <VMUtils/log.hpp>
#include <VMUtils/vmnew.hpp>
#include <VMFoundation/pluginloader.h>
#include <VMFoundation/logger.h>
#include <fstream>
#include <filesystem>

namespace vm
{
class BlockedGridVolumeFile__pImpl
{
	VM_DECL_API( BlockedGridVolumeFile )
public:
	BlockedGridVolumeFile__pImpl( BlockedGridVolumeFile *api ) :
	  q_ptr( api ) {}

	std::unique_ptr<RawStream> rawStream;
	Size3 blockDimension;
	int blockSizeInLog = -1;
	Size3 pageCount;
	int padding = 0;
	bool exact[ 3 ] = { false, false, false };
	Vec3i sampleStart;
	Vec3i blockNoPadding;
	std::unique_ptr<char[]> buf;  // buffer for a block

	bool IsBoundaryBlock( const Vec3i &start );

	void Create()
	{
		auto _ = q_func();
		assert( blockSizeInLog >= 0 );
		assert( padding >= 0 );
		assert( ( 1 << blockSizeInLog ) - 2 * padding > 0 );

		const auto &p = padding;
		sampleStart = { -p, -p, -p };
		blockNoPadding = Vec3i( _->Get3DPageSize() ) - Vec3i( 2 * p, 2 * p, 2 * p );

		const auto dataDimension = rawStream->GetDimension();
		pageCount = Size3( vm::RoundUpDivide( dataDimension.x, blockNoPadding.x ),
						   RoundUpDivide( dataDimension.y, blockNoPadding.y ),
						   RoundUpDivide( dataDimension.z, blockNoPadding.z ) );

		exact[ 0 ] = dataDimension.x % blockNoPadding.x == 0;
		exact[ 1 ] = dataDimension.y % blockNoPadding.y == 0;
		exact[ 2 ] = dataDimension.z % blockNoPadding.z == 0;
		buf.reset( new char[ blockDimension.Prod() * rawStream->GetElementSize() ] );
	}
};
bool BlockedGridVolumeFile__pImpl::IsBoundaryBlock( const Vec3i &start )
{
	const auto _ = q_func();
	const auto maxPoint = start + Vec3i( _->Get3DPageSize() );
	const auto dataSize = _->GetDimension();
	if ( start.x < 0 ||
		 start.y < 0 ||
		 start.z < 0 ||
		 ( exact[ 0 ] == false && maxPoint.x > dataSize.x ) ||
		 ( exact[ 1 ] == false && maxPoint.y > dataSize.y ) ||
		 ( exact[ 2 ] == false && maxPoint.z > dataSize.z ) ) return true;
	return false;
}

BlockedGridVolumeFile::BlockedGridVolumeFile( IRefCnt *cnt,
											  const std::string &fileName,
											  const vm::Size3 &dimensions,
											  size_t voxelSize,
											  int blockDimensionInLog, int padding ) :
  d_ptr( new BlockedGridVolumeFile__pImpl( this ) ),
  EverythingBase<I3DBlockFilePluginInterface>( cnt )
{
	VM_IMPL( BlockedGridVolumeFile )
	_->blockDimension = Size3( 1 << blockDimensionInLog, 1 << blockDimensionInLog, 1 << blockDimensionInLog );
	_->blockSizeInLog = blockDimensionInLog;
	_->rawStream = std::make_unique<RawStream>( fileName, dimensions, voxelSize );
	_->padding = padding;
	_->Create();
}

BlockedGridVolumeFile::BlockedGridVolumeFile( IRefCnt *cnt ) :
  EverythingBase<I3DBlockFilePluginInterface>( cnt ), d_ptr( new BlockedGridVolumeFile__pImpl( this ) )
{
}

void BlockedGridVolumeFile::Open( const std::string &fileName )
{
	VM_IMPL( BlockedGridVolumeFile )

	std::ifstream vifo( fileName );

	if ( vifo.is_open() == false ) {
		throw std::runtime_error( "Failed to open .brv file" );
	}

	std::string rawFileName;
	int x, y, z;
	int blockSizeInLog;

	vifo >> rawFileName >> x >> y >> z >> blockSizeInLog;

	std::filesystem::path pa( fileName );

	pa.replace_filename( rawFileName );
	_->blockDimension = Size3( 1 << blockSizeInLog, 1 << blockSizeInLog, 1 << blockSizeInLog );

	_->blockSizeInLog = blockSizeInLog;
	_->padding = 2;
	_->rawStream = std::make_unique<RawStream>( pa.string(), Size3( x, y, z ), 1 );

	_->Create();
}

bool BlockedGridVolumeFile::Create( const Block3DDataFileDesc *desc )
{
	LOG_DEBUG << "BlockedGirdVolumeFile::Create -- Not implement yet";
	return false;
}

void BlockedGridVolumeFile::Close()
{
	VM_IMPL( BlockedGridVolumeFile );
	_->rawStream = nullptr;
}

int BlockedGridVolumeFile::GetPadding() const
{
	const auto _ = d_func();
	return _->padding;
}

Size3 BlockedGridVolumeFile::GetDataSizeWithoutPadding() const
{
	const auto _ = d_func();
	return Size3( _->rawStream->GetDimension() );
}

Size3 BlockedGridVolumeFile::Get3DPageSize() const
{
	const auto _ = d_func();
	return _->blockDimension;
}

int BlockedGridVolumeFile::Get3DPageSizeInLog() const
{
	const auto _ = d_func();
	return _->blockSizeInLog;
}

size_t BlockedGridVolumeFile::GetPhysicalPageCount() const
{
	return 1;
}

size_t BlockedGridVolumeFile::GetVirtualPageCount() const
{
	return Get3DPageCount().Prod();
}

size_t BlockedGridVolumeFile::GetPageSize() const
{
	return Get3DPageSize().Prod();
}

vm::Size3 BlockedGridVolumeFile::Get3DPageCount() const
{
	const auto _ = d_func();
	return _->pageCount;
}

const void *BlockedGridVolumeFile::GetPage( size_t pageID )
{
	VM_IMPL( BlockedGridVolumeFile )
	// read boundary
	const auto idx3d = vm::Dim( pageID, { _->pageCount.x, _->pageCount.y } );
	const auto start = _->sampleStart + Vec3i( idx3d.x * _->blockNoPadding.x, idx3d.y * _->blockNoPadding.y, idx3d.z * _->blockNoPadding.z );
	if ( _->IsBoundaryBlock( start ) ) {
		_->rawStream->ReadRegionNoBoundary( start,
											_->blockDimension, (unsigned char *)_->buf.get() );
		return _->buf.get();
	} else {
		_->rawStream->ReadRegion( start,
								  _->blockDimension, (unsigned char *)_->buf.get() );
		return _->buf.get();
	}
	return nullptr;
}

inline void BlockedGridVolumeFile::Flush()
{
}

inline void BlockedGridVolumeFile::Write( const void *page, size_t pageID, bool flush )
{
	VM_IMPL( BlockedGridVolumeFile )
	// read boundary
	const auto idx3d = vm::Dim( pageID, { _->pageCount.x, _->pageCount.y } );
	const auto start = _->sampleStart + Vec3i( idx3d.x * _->blockNoPadding.x, idx3d.y * _->blockNoPadding.y, idx3d.z * _->blockNoPadding.z );
	if ( _->IsBoundaryBlock( start ) ) {
		_->rawStream->WriteRegionNoBoundary( start,
											 _->blockDimension, (const unsigned char *)page );
	} else {
		_->rawStream->WriteRegion( start,
								   _->blockDimension, (const unsigned char *)page );
	}
}

inline void BlockedGridVolumeFile::Flush( size_t pageID )
{
}

Vec3i BlockedGridVolumeFile::GetDimension() const
{
	const auto _ = d_func();
	return _->rawStream->GetDimension();
}

size_t BlockedGridVolumeFile::GetElementSize() const
{
	const auto _ = d_func();
	return _->rawStream->GetElementSize();
}

size_t BlockedGridVolumeFile::ReadRegion( const Vec3i &start, const Size3 &size, unsigned char *buffer )
{
	const auto _ = d_func();
	return _->rawStream->ReadRegion( start, size, buffer );
}

size_t BlockedGridVolumeFile::ReadRegionNoBoundary( const Vec3i &start, const Size3 &size, unsigned char *buffer )
{
	const auto _ = d_func();
	return _->rawStream->ReadRegionNoBoundary( start, size, buffer );
}

BlockedGridVolumeFile::~BlockedGridVolumeFile()
{
}

int BlockedGridVolumeFilePluginFactory::Keys( const char **keys ) const
{
	static const char *k[] = { ".brv" };
	return 1;
}

IEverything *BlockedGridVolumeFilePluginFactory::Create( const char *key )
{
	if ( std::strcmp( key, ".brv" ) == 0 ) {
		return VM_NEW<BlockedGridVolumeFile>();
	}
	return nullptr;
}

VM_REGISTER_PLUGIN_FACTORY_IMPL( BlockedGridVolumeFilePluginFactory )

VM_REGISTER_INTERNAL_PLUGIN_IMPL( BlockedGridVolumeFilePluginFactory )

}  // namespace vm
