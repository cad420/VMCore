
#include <VMFoundation/gridvolumefile.h>
#include <VMat/numeric.h>
#include <VMFoundation/rawreader.h>
#include <VMUtils/log.hpp>

#include <filesystem>

namespace vm
{
void VolumeFile::Create()
{
	const auto dataDimension = rawReader->GetDimension();
	pageCount = Size3( vm::RoundUpDivide( dataDimension.x, blockDimension.x ),
					   RoundUpDivide( dataDimension.y, blockDimension.y ),
					   RoundUpDivide( dataDimension.z, blockDimension.z ) );
	exact = ( dataDimension.x % blockDimension.x == 0 ) && ( dataDimension.y % blockDimension.y == 0 ) && ( dataDimension.z % blockDimension.z == 0 );

	buf.reset( new char[ dataDimension.Prod() * rawReader->GetElementSize() ] );
}

VolumeFile::VolumeFile( IRefCnt *cnt,
						const std::string &fileName,
						const vm::Size3 &dimensions,
						size_t voxelSize,
						int blockDimensionInLog ) :
  EverythingBase<I3DBlockFilePluginInterface>( cnt ),
  blockDimension( 1 << blockDimensionInLog, 1 << blockDimensionInLog, 1 << blockDimensionInLog ),
  blockSizeInLog( blockDimensionInLog )
{
	rawReader = std::make_unique<RawReader>( fileName, dimensions, voxelSize );
	Create();
}

VolumeFile::VolumeFile( IRefCnt *cnt ) :
  EverythingBase<I3DBlockFilePluginInterface>( cnt )
{
}

void VolumeFile::Open( const std::string &fileName )
{
	// a .vifo file
	std::ifstream vifo( fileName );
	if ( vifo.is_open() == false ) {
		throw std::runtime_error( "Failed to open .vifo file" );
	}

	std::string rawFileName;
	int x, y, z;
	int voxelSize;

	vifo >> rawFileName >> x >> y >> z >> voxelSize;

	std::filesystem::path pa( fileName );

	pa.replace_filename( rawFileName );
	vm::Debug( "Failed to open file: {}", pa.c_str() );
	blockDimension = Size3( x, y, z );
	rawReader = std::make_unique<RawReader>( fileName, blockDimension, voxelSize );

	blockSizeInLog = 6;  // 64 x 64 x 64 for a block
	Create();
}

int VolumeFile::GetPadding() const
{
	return padding;
}

Size3 VolumeFile::GetDataSizeWithoutPadding() const
{
	return Size3( rawReader->GetDimension() );
}

Size3 VolumeFile::Get3DPageSize() const
{
	return blockDimension * rawReader->GetElementSize();
}

int VolumeFile::Get3DPageSizeInLog() const
{
	return blockSizeInLog;
}

vm::Size3 VolumeFile::Get3DPageCount() const
{
	return pageCount;
}

const void *VolumeFile::GetPage( size_t pageID )
{
	// read boundary
	if ( !exact ) {
		const auto idx3d = vm::Dim( pageID, { pageCount.x, pageCount.y } );
		rawReader->readRegionNoBoundary( Vec3i( idx3d.x * blockDimension.x, idx3d.y * blockDimension.y, idx3d.z * blockDimension.z ),
										 blockDimension, (unsigned char *)buf.get() );
	} else {
		const auto idx3d = vm::Dim( pageID, { pageCount.x, pageCount.y } );
		rawReader->readRegion( Vec3i( idx3d.x * blockDimension.x, idx3d.y * blockDimension.y, idx3d.z * blockDimension.z ),
							   blockDimension, (unsigned char *)buf.get() );
	}
	return nullptr;
}

Vec3i VolumeFile::GetDimension() const
{
	return rawReader->GetDimension();
}

size_t VolumeFile::GetElementSize() const
{
	return rawReader->GetElementSize();
}

size_t VolumeFile::ReadRegion( const Vec3i &start, const Size3 &size, unsigned char *buffer )
{
	return rawReader->readRegion( start, size, buffer );
}

size_t VolumeFile::ReadRegionNoBoundary( const Vec3i &start, const Size3 &size, unsigned char *buffer )
{
	return rawReader->readRegionNoBoundary( start, size, buffer );
}
}