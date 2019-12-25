
#include <VMFoundation/blockedgridvolumefile.h>
#include <VMat/numeric.h>
#include <VMFoundation/rawreader.h>
#include <VMUtils/log.hpp>
#include <VMUtils/vmnew.hpp>
#include <VMFoundation/pluginloader.h>
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

	std::unique_ptr<RawReader> rawReader;
	Size3 blockDimension;
	int blockSizeInLog = -1;
	Size3 pageCount;
	int padding = 0;
	bool exact[ 3 ] = { false, false, false };
	Vec3i sampleStart;
	Vec3i blockNoPadding;
	//void Create();
	std::unique_ptr<char[]> buf;  // buffer for a block

	//bool IsBoundaryBlock( const Point3i &idx3d );
	bool IsBoundaryBlock( const Vec3i &start );
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

void BlockedGridVolumeFile::Create()
{
	VM_IMPL( BlockedGridVolumeFile )

	assert( _->blockSizeInLog >= 0 );
	assert( _->padding >= 0 );
	assert( ( 1 << _->blockSizeInLog ) - 2 * _->padding > 0 );

	const auto &p = _->padding;
	_->sampleStart = { -p, -p, -p };
	_->blockNoPadding = Vec3i( Get3DPageSize() ) - Vec3i( 2 * p, 2 * p, 2 * p );

	const auto dataDimension = _->rawReader->GetDimension();
	_->pageCount = Size3( vm::RoundUpDivide( dataDimension.x, _->blockNoPadding.x ),
						  RoundUpDivide( dataDimension.y, _->blockNoPadding.y ),
						  RoundUpDivide( dataDimension.z, _->blockNoPadding.z ) );

	_->exact[ 0 ] = dataDimension.x % _->blockNoPadding.x == 0;
	_->exact[ 1 ] = dataDimension.y % _->blockNoPadding.y == 0;
	_->exact[ 2 ] = dataDimension.z % _->blockNoPadding.z == 0;
	_->buf.reset( new char[ _->blockDimension.Prod() * _->rawReader->GetElementSize() ] );
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
	_->rawReader = std::make_unique<RawReader>( fileName, dimensions, voxelSize );
	_->padding = padding;
	Create();
}

BlockedGridVolumeFile::BlockedGridVolumeFile( IRefCnt *cnt ) :
  EverythingBase<I3DBlockFilePluginInterface>( cnt ),d_ptr( new BlockedGridVolumeFile__pImpl(this) )
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
	_->blockDimension = Size3(1<<blockSizeInLog,1<<blockSizeInLog,1<<blockSizeInLog);
	
	_->blockSizeInLog = blockSizeInLog;
	_->padding = 2;
	_->rawReader = std::make_unique<RawReader>( pa.string(), Size3(x,y,z), 1 );

	Create();
}

int BlockedGridVolumeFile::GetPadding() const
{
	const auto _ = d_func();
	return _->padding;
}

Size3 BlockedGridVolumeFile::GetDataSizeWithoutPadding() const
{
	const auto _ = d_func();
	return Size3( _->rawReader->GetDimension() );
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
	return Get3DPageCount().Prod();
}

size_t BlockedGridVolumeFile::GetVirtualPageCount() const
{
	return 1;
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
		_->rawReader->readRegionNoBoundary( start,
											_->blockDimension, (unsigned char *)_->buf.get() );
		return _->buf.get();
	} else {
		_->rawReader->readRegion( start,
								  _->blockDimension, (unsigned char *)_->buf.get() );
		return _->buf.get();
	}
	return nullptr;
}

Vec3i BlockedGridVolumeFile::GetDimension() const
{
	const auto _ = d_func();
	return _->rawReader->GetDimension();
}

size_t BlockedGridVolumeFile::GetElementSize() const
{
	const auto _ = d_func();
	return _->rawReader->GetElementSize();
}

size_t BlockedGridVolumeFile::ReadRegion( const Vec3i &start, const Size3 &size, unsigned char *buffer )
{
	const auto _ = d_func();
	return _->rawReader->readRegion( start, size, buffer );
}

size_t BlockedGridVolumeFile::ReadRegionNoBoundary( const Vec3i &start, const Size3 &size, unsigned char *buffer )
{
	const auto _ = d_func();
	return _->rawReader->readRegionNoBoundary( start, size, buffer );
}

BlockedGridVolumeFile::~BlockedGridVolumeFile()
{
}

std::vector<std::string> BlockedGridVolumeFilePluginFactory::Keys() const
{
	return { ".brv" };
}

IEverything *BlockedGridVolumeFilePluginFactory::Create( const std::string &key )
{
	if ( key == ".brv" ) {
		return VM_NEW<BlockedGridVolumeFile>();
	}
	return nullptr;
}

VM_REGISTER_PLUGIN_FACTORY_IMPL( BlockedGridVolumeFilePluginFactory )

VM_REGISTER_INTERNAL_PLUGIN_IMPL( BlockedGridVolumeFilePluginFactory )

}  // namespace vm
