
#include <VMFoundation/rawreader.h>
#include <VMCoreExtension/ifilemappingplugininterface.h>
#include <VMFoundation/libraryloader.h>
#include <VMFoundation/pluginloader.h>
#include <VMUtils/log.hpp>
#include <filesystem>

#include <cstring>  // memcpy
#include <cassert>
#include <iostream>
#include "VMat/numeric.h"

namespace vm
{
RawReader::RawReader( const std::string &fileName, const vm::Size3 &dimensions, size_t voxelSize ) :
  fileName( fileName ),
  dimensions( dimensions ),
  voxelSize( voxelSize )  //,file(fileName,std::ios::binary),
  ,
  offset( 0 ),
  ptr( nullptr )
{
	const auto rawBytes = dimensions.x * dimensions.y * dimensions.z * voxelSize;
	auto repo = LibraryReposity::GetLibraryRepo();
	assert( repo );
	repo->AddLibrary( "ioplugin" );

#ifdef _WIN32
	io = PluginLoader::GetPluginLoader()->CreatePlugin<IFileMapping>( "windows" );
#elif defined( __linux__ )
	io = PluginLoader::GetPluginLoader()->CreatePlugin<IFileMapping>( "linux" );
#endif
	if ( io == nullptr )
		throw std::runtime_error( "can not load ioplugin" );
	io->Open( fileName, rawBytes, FileAccess::Read, MapAccess::ReadOnly );
	ptr = io->FileMemPointer( 0, rawBytes );
	if ( !ptr ) {
		throw std::runtime_error( "map file failed RawReader::RawReader" );
	}
}

RawReader::~RawReader()
{
	io->DestroyFileMemPointer( ptr );
	io->Close();
}
// Read a region of volume data from the file into the buffer passed. It's assumed
// the buffer passed has enough room. Returns the number voxels read
size_t RawReader::readRegion( const vm::Size3 &start, const vm::Size3 &size, unsigned char *buffer )
{
	seekAmt = 0;
	return readRegion__( start, size, buffer );
}

std::size_t RawReader::readRegion__( const vm::Size3 &start, const vm::Size3 &size, unsigned char *buffer )
{
	assert( size.x > 0 && size.y > 0 && size.z > 0 );
	const uint64_t startRead = ( start.x + dimensions.x * ( start.y + dimensions.y * start.z ) ) * voxelSize;
	if ( offset != startRead ) {
		/*	seekAmt += startRead - offset;
				std::cout << seekAmt<< std::endl;

				seekAmt = startRead - offset;
				if (!file.seekg(seekAmt, std::ios_base::cur))
				{
					throw std::runtime_error("ImportRAW: Error seeking file");
				}

				if (fseek(file, seekAmt, SEEK_CUR) != 0)
				{
					throw std::runtime_error("ImportRAW: Error seeking file");
				}*/

		offset = startRead;
	}

	// Figure out how to actually read the region since it may not be a full X/Y slice and
	// we'll need to read the portions in X & Y and seek around to skip regions
	size_t read = 0;
	if ( convexRead( size ) )  // continuous read
	{
		//read = fread(buffer, voxelSize, size.x * size.y * size.z, file);

		//file.read(reinterpret_cast<char*>(buffer), voxelSize* size.x * size.y * size.z);
		//read = file.gcount()/voxelSize;

		memcpy( reinterpret_cast<void *>( buffer ), ptr + offset, voxelSize * size.x * size.y * size.z );

		read = size.x * size.y * size.z;  // voxel count

		offset = startRead + read * voxelSize;
	} else if ( size.x == dimensions.x ) {  // read by slice
		for ( auto z = start.z; z < start.z + size.z; ++z ) {
			const vm::Size3 startSlice( start.x, start.y, z );
			const vm::Size3 sizeSlice( size.x, size.y, 1 );
			read += readRegion__( startSlice, sizeSlice, buffer + read * voxelSize );
		}
	} else {
		for ( auto z = start.z; z < start.z + size.z; ++z ) {
			for ( auto y = start.y; y < start.y + size.y; ++y ) {
				const vm::Size3 startLine( start.x, y, z );
				const vm::Size3 sizeLine( size.x, 1, 1 );
				read += readRegion__( startLine, sizeLine, buffer + read * voxelSize );
			}
		}
	}
	return read;
}

RawReaderIO::RawReaderIO( const std::string &fileName, const vm::Size3 &dimensions, size_t voxelSize ) :
  fileName( fileName ),
  dimensions( dimensions ),
  voxelSize( voxelSize )  //,file(fileName,std::ios::binary),
  ,
  offset( 0 ),
  ptr( nullptr )
{
	const auto rawBytes = dimensions.x * dimensions.y * dimensions.z * voxelSize;
	file.open( fileName, std::ios::binary );
	if ( !file.is_open() ) {
		throw std::runtime_error( "RawReaderIO::RawReaderIO()::can not open file" );
	}
}

RawReaderIO::~RawReaderIO()
{
}

size_t RawReaderIO::readRegion( const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer )
{
	seekAmt = 0;
	return readRegion__( start, size, buffer );
}

size_t RawReaderIO::readRegionNoBoundary( const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer )
{
	return readRegionNoBoundary__( start, size, buffer );
}

std::size_t RawReaderIO::readRegion__( const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer )
{
	assert( size.x > 0 && size.y > 0 && size.z > 0 );
	const uint64_t startRead = ( start.x + dimensions.x * ( start.y + dimensions.y * start.z ) ) * voxelSize;
	if ( offset != startRead ) {
		seekAmt += startRead - offset;
		seekAmt = startRead - offset;
		if ( !file.seekg( seekAmt, std::ios_base::cur ) ) {
			throw std::runtime_error( "ImportRAW: Error seeking file" );
		}
		offset = startRead;
	}

	// Figure out how to actually read the region since it may not be a full X/Y slice and
	// we'll need to read the portions in X & Y and seek around to skip regions
	size_t read = 0;
	if ( convexRead( size ) )  // continuous read
	{
		//read = fread(buffer, voxelSize, size.x * size.y * size.z, file);

		file.read( reinterpret_cast<char *>( buffer ), voxelSize * size.x * size.y * size.z );
		//read = file.gcount() / voxelSize;

		//memcpy(reinterpret_cast<void*>(buffer), ptr + offset, voxelSize* size.x * size.y * size.z);

		read = size.x * size.y * size.z;  // voxel count

		offset = startRead + read * voxelSize;
	} else if ( size.x == dimensions.x ) {  // read by slice
		for ( auto z = start.z; z < start.z + size.z; ++z ) {
			const vm::Vec3i startSlice( start.x, start.y, z );
			const vm::Size3 sizeSlice( size.x, size.y, 1 );
			read += readRegion__( startSlice, sizeSlice, buffer + read * voxelSize );
		}
	} else {
		for ( auto z = start.z; z < start.z + size.z; ++z ) {
			for ( auto y = start.y; y < start.y + size.y; ++y ) {
				const vm::Vec3i startLine( start.x, y, z );
				const vm::Size3 sizeLine( size.x, 1, 1 );
				read += readRegion__( startLine, sizeLine, buffer + read * voxelSize );
			}
		}
	}
	return read;
}

size_t RawReaderIO::readRegionNoBoundary__( const vm::Vec3i &start, const vm::Size3 &size, unsigned char * buffer )
{
	assert( size.x > 0 && size.y > 0 && size.z > 0 );

	const uint64_t startRead = ( start.x + dimensions.x * ( start.y + dimensions.y * start.z ) ) * voxelSize;
	if ( offset != startRead ) {
		seekAmt += startRead - offset;
		seekAmt = startRead - offset;
		if ( !file.seekg( seekAmt, std::ios_base::cur ) ) {
			throw std::runtime_error( "ImportRAW: Error seeking file" );
		}
		offset = startRead;
	}

	// Figure out how to actually read the region since it may not be a full X/Y slice and
	// we'll need to read the portions in X & Y and seek around to skip regions
	size_t read = 0;
	if ( convexReadNoBoundary( start, size ) )  // continuous read
	{
		size_t read__ = 0;
		const auto xlSize = std::max( 0 - start.x, 0 );
		const auto ybSize = std::max( 0 - start.y, 0 );
		const auto zdSize = std::max( 0 - start.z, 0 );

		const auto xrSize = std::max( 0, int(start.x + size.x - dimensions.x ));
		const auto yfSize = std::max( 0, int(start.y + size.y - dimensions.y ));
		const auto zuSize = std::max( 0, int(start.z + size.z - dimensions.z ));

		const auto xmSize = size.x - xlSize - xrSize;
		const auto ymSize = size.y - ybSize - yfSize;
		const auto zmSize = size.z - zdSize - zuSize;
		

		memset( buffer + read__,0, size_t( xlSize ) * ybSize * zdSize * voxelSize );
		read__ += size_t( xlSize ) * ybSize * zdSize;

		file.read( reinterpret_cast<char *>( buffer + read__ ), voxelSize * xmSize * ymSize * zmSize );
		read__ += xmSize  * ymSize * zmSize;

		memset( buffer + read__, 0, size_t( xrSize ) * yfSize * zmSize );
		read__ += size_t( xrSize ) * yfSize * zmSize;

		read = size.x * size.y * size.z;  // voxel count

		assert( read__ == read );

		offset = startRead + read * voxelSize;
	} else if ( start.x + size.x == dimensions.x ) {  // read by slice
		for ( auto z = start.z; z < start.z + size.z; ++z ) {
			const vm::Vec3i startSlice( start.x, start.y, z );
			const vm::Size3 sizeSlice( size.x, size.y, 1 );
			read += readRegionNoBoundary__( startSlice, sizeSlice, buffer + read * voxelSize );
		}
	} else {
		for ( auto z = start.z; z < start.z + size.z; ++z ) {
			for ( auto y = start.y; y < start.y + size.y; ++y ) {
				const vm::Vec3i startLine( start.x, y, z );
				const vm::Size3 sizeLine( size.x, 1, 1 );
				read += readRegionNoBoundary__( startLine, sizeLine, buffer + read * voxelSize );
			}
		}
	}
	return read;
}

void RawFile::Create()
{
	const auto dataDimension = rawReader->GetDimension();
	pageCount = Size3( vm::RoundUpDivide( dataDimension.x, blockDimension.x ),
					   RoundUpDivide( dataDimension.y, blockDimension.y ),
					   RoundUpDivide( dataDimension.z, blockDimension.z ) );
	exact = (dataDimension.x % blockDimension.x == 0)
	&& (dataDimension.y % blockDimension.y ==0)
	&& (dataDimension.z % blockDimension.z == 0);

	buf.reset( new char[ dataDimension.Prod() * rawReader->GetElementSize() ] );
}

RawFile::RawFile( IRefCnt *cnt,
				  const std::string &fileName,
				  const vm::Size3 &dimensions,
				  size_t voxelSize,
				  int blockDimensionInLog ) :
  EverythingBase<I3DBlockFilePluginInterface>( cnt ),
  blockDimension( 1 << blockDimensionInLog, 1 << blockDimensionInLog, 1 << blockDimensionInLog ),
  blockSizeInLog( blockDimensionInLog )
{
	rawReader = std::make_unique<RawReaderIO>( fileName, dimensions, voxelSize );
	Create();
}

RawFile::RawFile( IRefCnt *cnt ) :
  EverythingBase<I3DBlockFilePluginInterface>( cnt )
{
}

void RawFile::Open( const std::string &fileName )
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
	rawReader = std::make_unique<RawReaderIO>( fileName, blockDimension, voxelSize );

	blockSizeInLog = 6;  // 64 x 64 x 64 for a block
	Create();
}

int RawFile::GetPadding() const
{
	return padding;
}

Size3 RawFile::GetDataSizeWithoutPadding() const
{
	return rawReader->GetDimension();
}

Size3 RawFile::Get3DPageSize() const
{
	return blockDimension * rawReader->GetElementSize();
}

int RawFile::Get3DPageSizeInLog() const
{
	return blockSizeInLog;
}

vm::Size3 RawFile::Get3DPageCount() const
{
	return pageCount;
}

const void *RawFile::GetPage( size_t pageID )
{
	// read boundary
	if ( !exact ) {
		const auto idx3d = vm::Dim( pageID, { pageCount.x, pageCount.y } );
		rawReader->readRegionNoBoundary( Vec3i( idx3d.x * blockDimension.x, idx3d.y * blockDimension.y, idx3d.z * blockDimension.z ),
							   blockDimension, (unsigned char *)buf.get() );
	} else {
	    const auto idx3d = vm::Dim( pageID, { pageCount.x, pageCount.y } );
		rawReader->readRegion( Vec3i(idx3d.x * blockDimension.x,idx3d.y*blockDimension.y ,idx3d.z * blockDimension.z ),
			blockDimension, (unsigned char *)buf.get() );
	}
	return nullptr;
}

Size3 RawFile::GetDimension() const
{
	return rawReader->GetDimension();
}

size_t RawFile::GetElementSize() const
{
	return rawReader->GetElementSize();
}

size_t RawFile::ReadRegion( const Vec3i &start, const Size3 &size, unsigned char *buffer )
{
	return rawReader->readRegion( start, size, buffer );
}

size_t RawFile::ReadRegionNoBoundary( const Vec3i &start, const Size3 &size, unsigned char *buffer )
{
	return rawReader->readRegionNoBoundary( start, size, buffer );
}
}  // namespace vm
