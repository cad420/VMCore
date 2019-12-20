
#include <VMFoundation/rawreader.h>
#include <VMCoreExtension/ifilemappingplugininterface.h>
#include <VMFoundation/libraryloader.h>
#include <VMFoundation/pluginloader.h>
#include <VMUtils/log.hpp>
#include <VMat/geometry.h>
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
	const Bound3i dataBound( { 0, 0, 0 }, Point3i( dimensions.x, dimensions.y, dimensions.z ) );
	const Bound3i readBound( { start.x, start.y, start.z }, Point3i( size.x, size.y, size.z ) + start );
	const auto isectBound = dataBound.IntersectWidth( readBound );
	const auto dstSize = Size3(readBound.Diagonal());
	const auto dig = isectBound.Diagonal();
	const auto readSize = Size3( dig.x, dig.y, dig.z );
	auto buf = stagingBuffer.Alloc<unsigned char>( isectBound.Volume(), true );
	const auto read = readRegion( { isectBound.min.x, isectBound.min.y, isectBound.min.z }, readSize, buf );
	assert( read == readSize.Prod() );
	const auto newStart = isectBound.min - readBound.min;

	std::function<size_t( unsigned char *,
						const Vec3i &,
						const unsigned char *, 
		                const Size3 & )>
	  carray = [&carray,this,&dstSize](unsigned char * dst,
		  		  const Vec3i & start,
		  const unsigned char * src,
		  const Size3 & srcSize)
	{
		  //assert( size.x > 0 && size.y > 0 && size.z > 0 );
		  // Figure out how to actually read the region since it may not be a full X/Y slice and
		  // we'll need to read the portions in X & Y and seek around to skip regions
		  size_t r = 0;
		  if ( 
			  ( srcSize.x == dstSize.x && srcSize.y == dstSize.y ) 
			  || ( srcSize.x == dstSize.x && srcSize.z == 1 ) 
			  || ( srcSize.y == 1 && srcSize.z == 1 )
			  )  // continuous read
		  {
			  const uint64_t offset = ( start.x + dstSize.x * ( start.y + dstSize.y * start.z ) ) * voxelSize;
			  memcpy( dst + offset, src, voxelSize * srcSize.x * srcSize.y * srcSize.z );
			  r = srcSize.x * srcSize.y * srcSize.z;  // voxel count
		  } else if ( srcSize.x == dstSize.x ) {	  // read by slice
			  for ( auto z = start.z; z < start.z + srcSize.z; ++z ) {
				  const Vec3i startSlice( start.x, start.y, z );
				  const Size3 sizeSlice( srcSize.x, srcSize.y, 1 );
				  r += carray( dst, startSlice, src + r * voxelSize, sizeSlice );
			  }
		  } else {
			  for ( auto z = start.z; z < start.z + srcSize.z; ++z ) {
				  for ( auto y = start.y; y < start.y + srcSize.y; ++y ) {
					  const Vec3i startLine( start.x, y, z );
					  const Size3 sizeLine( srcSize.x, 1, 1 );
					  r += carray( dst, startLine, src + r * voxelSize,sizeLine);
				  }
			  }
		  }
		  return r;
	};

	return carray( buffer, newStart, buf, readSize );
}

size_t RawReaderIO::Transport3D( const unsigned char *src, const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer )
{
	//return transport3d__( src, start, size, buffer );
	return 0;
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

//size_t RawReaderIO::transport3d__( unsigned char *dst,
//								   const vm::Vec3i &start,
//								   const vm::Size3 &size,
//								   const unsigned char *src,
//								   const Vec3i &srcSize )
//{
//	assert( size.x > 0 && size.y > 0 && size.z > 0 );
//	// Figure out how to actually read the region since it may not be a full X/Y slice and
//	// we'll need to read the portions in X & Y and seek around to skip regions
//
//	size_t read = 0;
//	if ( convexRead( size ) )  // continuous read
//	{
//		const uint64_t startRead = ( start.x + dimensions.x * ( start.y + dimensions.y * start.z ) ) * voxelSize;
//		memcpy( dst + startRead, src, voxelSize * size.x * size.y * size.z );
//		read = size.x * size.y * size.z;	// voxel count
//	} else if ( size.x == dimensions.x ) {  // read by slice
//		for ( auto z = start.z; z < start.z + size.z; ++z ) {
//			const vm::Vec3i startSlice( start.x, start.y, z );
//			const vm::Size3 sizeSlice( size.x, size.y, 1 );
//			read += transport3d__( src + read * voxelSize, startSlice, sizeSlice, dst );
//		}
//	} else {
//		for ( auto z = start.z; z < start.z + size.z; ++z ) {
//			for ( auto y = start.y; y < start.y + size.y; ++y ) {
//				const vm::Vec3i startLine( start.x, y, z );
//				const vm::Size3 sizeLine( size.x, 1, 1 );
//				read += transport3d__( src + read * voxelSize, startLine, sizeLine, dst );
//			}
//		}
//	}
//	return read;
//}

void RawFile::Create()
{
	const auto dataDimension = rawReader->GetDimension();
	pageCount = Size3( vm::RoundUpDivide( dataDimension.x, blockDimension.x ),
					   RoundUpDivide( dataDimension.y, blockDimension.y ),
					   RoundUpDivide( dataDimension.z, blockDimension.z ) );
	exact = ( dataDimension.x % blockDimension.x == 0 ) && ( dataDimension.y % blockDimension.y == 0 ) && ( dataDimension.z % blockDimension.z == 0 );

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
		rawReader->readRegion( Vec3i( idx3d.x * blockDimension.x, idx3d.y * blockDimension.y, idx3d.z * blockDimension.z ),
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
