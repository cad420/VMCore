
#include <VMFoundation/gridvolumefile.h>
#include <VMat/numeric.h>
#include <VMFoundation/rawreader.h>
#include <VMUtils/log.hpp>

#include <filesystem>

namespace vm
{


class GridVolumeFile__pImpl
{
	VM_DECL_API( GridVolumeFile )
public:

	GridVolumeFile__pImpl( GridVolumeFile *api ) :
	  q_ptr( api ) {}
	
	std::unique_ptr<RawReader> rawReader;
	Size3 blockDimension;
	int blockSizeInLog = -1;
	Size3 pageCount;
	const int padding = 0;
	bool exact = false;
	void Create();
	std::unique_ptr<char[]> buf;  // buffer for a block
};


void GridVolumeFile::Create()
{
	VM_IMPL( GridVolumeFile )
	
	const auto dataDimension = _->rawReader->GetDimension();
	_->pageCount = Size3( vm::RoundUpDivide( dataDimension.x, _->blockDimension.x ),
						  RoundUpDivide( dataDimension.y, _->blockDimension.y ),
						  RoundUpDivide( dataDimension.z, _->blockDimension.z ) );
	_->exact = ( dataDimension.x % _->blockDimension.x == 0 ) && ( dataDimension.y % _->blockDimension.y == 0 ) && ( dataDimension.z % _->blockDimension.z == 0 );

	_->buf.reset( new char[ dataDimension.Prod() * _->rawReader->GetElementSize() ] );
}

GridVolumeFile::GridVolumeFile( IRefCnt *cnt,
						const std::string &fileName,
						const vm::Size3 &dimensions,
						size_t voxelSize,
						int blockDimensionInLog ):
d_ptr( new GridVolumeFile__pImpl(this) ),
 EverythingBase<I3DBlockFilePluginInterface>( cnt )
{
	VM_IMPL( GridVolumeFile )
	_->blockDimension = Size3( 1 << blockDimensionInLog, 1 << blockDimensionInLog, 1 << blockDimensionInLog );
	_->blockSizeInLog = blockDimensionInLog;
	_->rawReader = std::make_unique<RawReader>( fileName, dimensions, voxelSize );
	Create();
}

GridVolumeFile::GridVolumeFile( IRefCnt *cnt ) :
  EverythingBase<I3DBlockFilePluginInterface>( cnt )
{
}

void GridVolumeFile::Open( const std::string &fileName )
{

	VM_IMPL( GridVolumeFile )
	
	// a .vifo file
	std::ifstream vifo( fileName );
	if ( vifo.is_open() == false ) 
	{
		throw std::runtime_error( "Failed to open .vifo file" );
	}

	std::string rawFileName;
	int x, y, z;
	int voxelSize;

	vifo >> rawFileName >> x >> y >> z >> voxelSize;

	std::filesystem::path pa( fileName );

	pa.replace_filename( rawFileName );
	vm::Debug( "Failed to open file: {}", pa.c_str() );
	_->blockDimension = Size3( x, y, z );
	_->rawReader = std::make_unique<RawReader>( fileName, _->blockDimension, voxelSize );

	_->blockSizeInLog = 6;  // 64 x 64 x 64 for a block
	Create();
}

int GridVolumeFile::GetPadding() const
{
	const auto _ = d_func();
	return _->padding;
}

Size3 GridVolumeFile::GetDataSizeWithoutPadding() const
{
	const auto _ = d_func();
	return Size3( _->rawReader->GetDimension() );
}

Size3 GridVolumeFile::Get3DPageSize() const
{
	const auto _ = d_func();
	return _->blockDimension * _->rawReader->GetElementSize();
}

int GridVolumeFile::Get3DPageSizeInLog() const
{
	const auto _ = d_func();
	return _->blockSizeInLog;
}

vm::Size3 GridVolumeFile::Get3DPageCount() const
{
	const auto _ = d_func();
	return _->pageCount;
}

const void *GridVolumeFile::GetPage( size_t pageID )
{
	VM_IMPL( GridVolumeFile )
	// read boundary
	if ( !_->exact ) {
		const auto idx3d = vm::Dim( pageID, { _->pageCount.x, _->pageCount.y } );
		_->rawReader->readRegionNoBoundary( Vec3i( idx3d.x * _->blockDimension.x, idx3d.y * _->blockDimension.y, idx3d.z * _->blockDimension.z ),
											_->blockDimension, (unsigned char *)_->buf.get() );
	} else {
		const auto idx3d = vm::Dim( pageID, { _->pageCount.x, _->pageCount.y } );
		_->rawReader->readRegion( Vec3i( idx3d.x * _->blockDimension.x, idx3d.y * _->blockDimension.y, idx3d.z * _->blockDimension.z ),
							   _->blockDimension, (unsigned char *)_->buf.get() );
	}
	return nullptr;
}

Vec3i GridVolumeFile::GetDimension() const
{
	const auto _ = d_func();
	return _->rawReader->GetDimension();
}

size_t GridVolumeFile::GetElementSize() const
{
	const auto _ = d_func();
	return _->rawReader->GetElementSize();
}

size_t GridVolumeFile::ReadRegion( const Vec3i &start, const Size3 &size, unsigned char *buffer )
{
	const auto _ = d_func();
	return _->rawReader->readRegion( start, size, buffer );
}

size_t GridVolumeFile::ReadRegionNoBoundary( const Vec3i &start, const Size3 &size, unsigned char *buffer )
{
	const auto _ = d_func();
	return _->rawReader->readRegionNoBoundary( start, size, buffer );
}
}