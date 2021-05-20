#include "VMFoundation/logger.h"
#include <VMFoundation/rawstream.h>
#include <VMCoreExtension/ifilemappingplugininterface.h>
#include <VMFoundation/libraryloader.h>
#include <VMFoundation/pluginloader.h>
#include <VMUtils/log.hpp>
#include <VMat/geometry.h>
#include <VMUtils/ref.hpp>
#include <VMFoundation/dataarena.h>

#include <cmath>
#include <fstream>
#include <filesystem>
#include <cstring>	// memcpy
#include <cassert>
#include <ios>
#include <stdexcept>

namespace vm
{
class RawStream__pImpl
{
	VM_DECL_API( RawStream )
public:
	RawStream__pImpl( RawStream *api ) :
	  q_ptr( api ) {}

	Bound3i dataBound;
	size_t voxelSize = 1;
	int64_t offset = 0;
	std::fstream file;
	unsigned char *ptr = nullptr;
	int64_t seekAmt = 0;
	std::function<size_t( const Vec3i &, const vm::Size3 &, unsigned char * )> readRegion;
	std::function<size_t( const Vec3i &, const vm::Size3 &, const unsigned char * )> writeRegion;

	void Close()
	{
		ptr = nullptr;
		seekAmt = 0;
		offset = 0;
		if ( file.is_open() ) {
			file.close();
		}
	}
};

namespace
{
size_t Linear2Volume( unsigned char *dst,
					  const Size3 &dstSize,
					  const Vec3i &start,
					  const unsigned char *src,
					  const Size3 &srcSize )
{
	size_t r = 0;
	if ( ( srcSize.x == dstSize.x && srcSize.y == dstSize.y ) || ( srcSize.x == dstSize.x && srcSize.z == 1 ) || ( srcSize.y == 1 && srcSize.z == 1 ) )	 // continuous read
	{
		const uint64_t offset = ( start.x + dstSize.x * ( start.y + dstSize.y * start.z ) );
		memcpy( dst + offset, src, srcSize.x * srcSize.y * srcSize.z );
		r = srcSize.x * srcSize.y * srcSize.z;	// voxel count
	} else if ( srcSize.x == dstSize.x ) {		// read by slice
		for ( auto z = start.z; z < start.z + srcSize.z; ++z ) {
			const Vec3i startSlice( start.x, start.y, z );
			const Size3 sizeSlice( srcSize.x, srcSize.y, 1 );
			r += Linear2Volume( dst, dstSize, startSlice, src + r, sizeSlice );
		}
	} else {
		for ( auto z = start.z; z < start.z + srcSize.z; ++z ) {
			for ( auto y = start.y; y < start.y + srcSize.y; ++y ) {
				const Vec3i startLine( start.x, y, z );
				const Size3 sizeLine( srcSize.x, 1, 1 );
				r += Linear2Volume( dst, dstSize, startLine, src + r, sizeLine );
			}
		}
	}
	return r;
}

size_t Volume2Linear( unsigned char *dst,
					  const unsigned char *src,
					  const Size3 &srcSize,
					  const Vec3i &start,
					  const Size3 &size )
{
	size_t r = 0;
	if ( ( srcSize.x == size.x && srcSize.y == size.y ) || ( srcSize.x == size.x && srcSize.z == 1 ) || ( size.y == 1 && size.z == 1 ) )  // continuous read
	{
		const uint64_t offset = ( start.x + srcSize.x * ( start.y + srcSize.y * start.z ) );
		memcpy( dst, src + offset, size.x * size.y * size.z );
		r = size.x * size.y * size.z;	 // voxel count
	} else if ( size.x == srcSize.x ) {	 // read by slice
		for ( auto z = start.z; z < start.z + size.z; ++z ) {
			const Vec3i startSlice( start.x, start.y, z );
			const Size3 sizeSlice( size.x, size.y, 1 );
			r += Volume2Linear( dst + r, src, srcSize, startSlice, sizeSlice );
		}
	} else {
		for ( auto z = start.z; z < start.z + size.z; ++z ) {
			for ( auto y = start.y; y < start.y + size.y; ++y ) {
				const Vec3i startLine( start.x, y, z );
				const Size3 sizeLine( size.x, 1, 1 );
				r += Volume2Linear( dst + r, src, srcSize, startLine, sizeLine );
			}
		}
	}
	return r;
}

}  // namespace

RawStream::RawStream( unsigned char *src, const Size3 &dimensions, size_t voxelSize ) :
  d_ptr( new RawStream__pImpl( this ) )
{
	assert( src );
	VM_IMPL( RawStream );
	_->dataBound = { { 0, 0, 0 }, { int( dimensions.x ), int( dimensions.y ), int( dimensions.z ) } };
	_->voxelSize = voxelSize;
	_->ptr = src;
	_->readRegion = [ this ]( const Vec3i &start,
							  const Size3 &size,
							  unsigned char *buffer ) {
		VM_IMPL( RawStream )
		return Volume2Linear( buffer, _->ptr, Size3( _->dataBound.Diagonal() ), start, size );
	};

	_->writeRegion = [ this ]( const Vec3i &start,
							   const Size3 &size,
							   const unsigned char *buffer ) {
		VM_IMPL( RawStream )
		return Linear2Volume( _->ptr, Size3( _->dataBound.Diagonal() ), start, buffer, size );
	};
}

RawStream::RawStream( const std::string &fileName,
					  const Size3 &dimensions,
					  size_t voxelSize ) :
  d_ptr( new RawStream__pImpl( this ) )
{
	VM_IMPL( RawStream );

	bool newCreated = false;
	auto cp = std::filesystem::current_path();
	auto filePath = cp / ( std::filesystem::path( fileName ) );
	// std::cout<<filePath;

	size_t expectedSize = dimensions.Prod() * voxelSize;
	if ( std::filesystem::exists( filePath ) ) {
		const auto filesize = std::filesystem::file_size( filePath );
		if ( filesize != expectedSize ) {
			LOG_DEBUG << "wrong file size: " << filesize << ", which " << expectedSize << " is expected.";
		}
	} else {
		newCreated = true;
		_->file.open( filePath, std::ios::out );  // You can create a new file with fstream::in
		_->file.close();
		std::filesystem::resize_file( filePath, expectedSize );
	}
	_->file.open( filePath, fstream::binary | fstream::in | fstream::out );
	if ( _->file.is_open() == false ) {
		throw std::runtime_error( "can not open file" );
	}

	_->dataBound = { { 0, 0, 0 }, { int( dimensions.x ), int( dimensions.y ), int( dimensions.z ) } };
	_->voxelSize = voxelSize;
	_->readRegion = [ this ]( const Vec3i &start,
							  const Size3 &size,
							  unsigned char *buffer ) {
		VM_IMPL( RawStream )
		_->seekAmt = 0;
		return ReadRegion__Implement( start, size, buffer );
	};

	_->writeRegion = [ this ]( const Vec3i &start,
							   const Size3 &size,
							   const unsigned char *buffer ) {
		VM_IMPL( RawStream )
		_->seekAmt = 0;
		return WriteRegion__Implement( start, size, buffer );
	};
}

bool RawStream::IsOpened() const
{
	const auto _ = d_func();
	return _->file.is_open();
}

RawStream::~RawStream()
{
}

size_t RawStream::ReadRegion( const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer )
{
	VM_IMPL( RawStream )
	return _->readRegion( start, size, buffer );
}

size_t RawStream::ReadRegionNoBoundary( const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer )
{
	VM_IMPL( RawStream )

	const Bound3i readBound( { start.x, start.y, start.z }, Point3i( size.x, size.y, size.z ) + start );
	if ( _->dataBound.InsideEx( readBound ) == true ) {
		return _->readRegion( start, size, buffer );
	}
	const auto isectBound = _->dataBound.IntersectWidth( readBound );
	if ( isectBound.IsNull() ) {
		memset( buffer, 0, size.Prod() * _->voxelSize );
		return size.Prod();
	}
	const auto dig = isectBound.Diagonal();
	const auto dstSize = Size3( readBound.Diagonal() );
	const auto readSize = Size3( dig.x, dig.y, dig.z );

	auto buf = (unsigned char *)calloc( _->voxelSize, readSize.Prod() );
	memset( buf, 0, readSize.Prod() * _->voxelSize );

	const auto read = ReadRegion( { isectBound.min.x, isectBound.min.y, isectBound.min.z }, readSize, buf );

	assert( read == readSize.Prod() );

	const auto newStart = isectBound.min - readBound.min;

	auto ret = Linear2Volume( buffer, dstSize * _->voxelSize, newStart * _->voxelSize, buf, readSize * _->voxelSize );

	free( buf );
	return ret;
}

size_t RawStream::WriteRegion( const vm::Vec3i &start,
							   const vm::Size3 &size, const unsigned char *buffer )
{
	VM_IMPL( RawStream )
	return _->writeRegion( start, size, buffer );
}

size_t RawStream::WriteRegionNoBoundary( const vm::Vec3i &start,
										 const vm::Size3 &size, const unsigned char *buffer )
{
	VM_IMPL( RawStream )

	const Bound3i writeBound( { start.x, start.y, start.z }, Point3i( size.x, size.y, size.z ) + start );
	if ( _->dataBound.InsideEx( writeBound ) == true ) {
		return _->writeRegion( start, size, buffer );
	}
	const auto isectBound = _->dataBound.IntersectWidth( writeBound );
	if ( isectBound.IsNull() ) {
		return 0;
	}
	const auto dig = isectBound.Diagonal();
	const auto srcSize = Size3( writeBound.Diagonal() );
	const auto writeSize = Size3( dig.x, dig.y, dig.z );

	auto buf = (unsigned char *)calloc( _->voxelSize, writeSize.Prod() );

	const auto newStart = isectBound.min - writeBound.min;
	Volume2Linear( buf, buffer, srcSize, newStart, writeSize );

	const auto write = WriteRegion( { isectBound.min.x, isectBound.min.y, isectBound.min.z }, writeSize, buf );
	assert( write == writeSize.Prod() );
	free( buf );
	return write;
}

Vec3i RawStream::GetDimension() const
{
	const auto _ = d_func();
	return _->dataBound.Diagonal();
}

size_t RawStream::GetElementSize() const
{
	const auto _ = d_func();
	return _->voxelSize;
}

std::size_t RawStream::ReadRegion__Implement( const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer )
{
	VM_IMPL( RawStream )
	assert( size.x > 0 && size.y > 0 && size.z > 0 );
	//const uint64_t startRead = ( start.x + _->dataBound.max.x * ( start.y + _->dataBound.max.y * start.z ) ) * _->voxelSize;
	const uint64_t startRead = Linear( start.ToPoint3(), Size2( _->dataBound.max.x, _->dataBound.max.y ) ) * _->voxelSize;
	if ( _->offset != startRead ) {
		_->seekAmt = startRead - _->offset;
		if ( !_->file.seekp( _->seekAmt, std::ios_base::cur ) ) {
			std::cout << "seekg failed";
			throw std::runtime_error( "RAW: Error seeking file" );
		}
		_->offset = startRead;
	}
	// Figure out how to actually read the region since it may not be a full X/Y slice and
	// we'll need to read the portions in X & Y and seek around to skip regions
	size_t read = 0;
	if ( IsConvex( size ) )	 // continuous read
	{
		read = size.x * size.y * size.z;  // voxel count
		const std::size_t readBytes = _->voxelSize * read;
		_->file.read( reinterpret_cast<char *>( buffer ), readBytes );

		_->offset = startRead + readBytes;
	} else if ( size.x == _->dataBound.max.x ) {  // read by slice
		for ( auto z = start.z; z < start.z + size.z; ++z ) {
			const vm::Vec3i startSlice( start.x, start.y, z );
			const vm::Size3 sizeSlice( size.x, size.y, 1 );
			read += ReadRegion__Implement( startSlice, sizeSlice, buffer + read * _->voxelSize );
		}
	} else {
		for ( auto z = start.z; z < start.z + size.z; ++z ) {
			for ( auto y = start.y; y < start.y + size.y; ++y ) {
				const vm::Vec3i startLine( start.x, y, z );
				const vm::Size3 sizeLine( size.x, 1, 1 );
				read += ReadRegion__Implement( startLine, sizeLine, buffer + read * _->voxelSize );
			}
		}
	}
	return read;
}

std::size_t RawStream::WriteRegion__Implement( const vm::Vec3i &start, const vm::Size3 &size, const unsigned char *buffer )
{
	VM_IMPL( RawStream )
	assert( size.x > 0 && size.y > 0 && size.z > 0 );
	const uint64_t startWrite = Linear( start.ToPoint3(), Size2( _->dataBound.max.x, _->dataBound.max.y ) ) * _->voxelSize;
	if ( _->offset != startWrite ) {
		_->seekAmt = startWrite - _->offset;
		if ( !_->file.seekp( _->seekAmt, std::ios_base::cur ) ) {
			std::cout << _->file.bad() << " " << _->file.eofbit << " " << _->file.failbit << std::endl;
			exit( -1 );
		}
		_->offset = startWrite;
	}
	size_t write = 0;
	if ( IsConvex( size ) )	 // continuous write
	{
		_->file.write( reinterpret_cast<const char *>( buffer ), _->voxelSize * size.x * size.y * size.z );
		write = size.x * size.y * size.z;  // voxel count
		_->offset = startWrite + write * _->voxelSize;
	} else if ( size.x == _->dataBound.max.x ) {  // write by slice
		for ( auto z = start.z; z < start.z + size.z; ++z ) {
			const vm::Vec3i startSlice( start.x, start.y, z );
			const vm::Size3 sizeSlice( size.x, size.y, 1 );
			write += WriteRegion__Implement( startSlice, sizeSlice, buffer + write * _->voxelSize );
		}
	} else {
		for ( auto z = start.z; z < start.z + size.z; ++z ) {
			for ( auto y = start.y; y < start.y + size.y; ++y ) {
				const vm::Vec3i startLine( start.x, y, z );
				const vm::Size3 sizeLine( size.x, 1, 1 );
				write += WriteRegion__Implement( startLine, sizeLine, buffer + write * _->voxelSize );
			}
		}
	}
	return write;
}

bool RawStream::IsConvex( const vm::Size3 &size ) const
{
	const auto _ = d_func();
	return ( size.x == _->dataBound.max.x && size.y == _->dataBound.max.y ) || ( size.x == _->dataBound.max.x && size.z == 1 ) || ( size.y == 1 && size.z == 1 );
}

}  // namespace vm
