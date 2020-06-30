
#include <VMFoundation/rawreader.h>
#include <VMCoreExtension/ifilemappingplugininterface.h>
#include <VMFoundation/libraryloader.h>
#include <VMFoundation/pluginloader.h>
#include <VMUtils/log.hpp>
#include <VMat/geometry.h>
#include <VMUtils/ref.hpp>
#include <VMFoundation/dataarena.h>

#include <fstream>
#include <filesystem>
#include <cstring>  // memcpy
#include <cassert>

namespace vm
{
class RawReader__pImpl
{
	VM_DECL_API( RawReader )
public:
	RawReader__pImpl( RawReader *api ) :
	  q_ptr( api ) {}

	Bound3i dataBound;
	size_t voxelSize = 1;
	int64_t offset = 0;
	std::ifstream file;
	unsigned char *ptr = nullptr;
	int64_t seekAmt = 0;
	DataArena<64> stagingBuffer = DataArena<64>(1024*1024*10);
	Ref<IMappingFile> mappingFile;
	std::function<size_t( const Vec3i &, const vm::Size3 &, unsigned char * )> readRegion;
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
	if ( ( srcSize.x == dstSize.x && srcSize.y == dstSize.y ) || ( srcSize.x == dstSize.x && srcSize.z == 1 ) || ( srcSize.y == 1 && srcSize.z == 1 ) )  // continuous read
	{
		const uint64_t offset = ( start.x + dstSize.x * ( start.y + dstSize.y * start.z ) );
		memcpy( dst + offset, src, srcSize.x * srcSize.y * srcSize.z );
		r = srcSize.x * srcSize.y * srcSize.z;  // voxel count
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
		r = size.x * size.y * size.z;	// voxel count
	} else if ( size.x == srcSize.x ) {  // read by slice
		for ( auto z = start.z; z < start.z + srcSize.z; ++z ) {
			const Vec3i startSlice( start.x, start.y, z );
			const Size3 sizeSlice( size.x, size.y, 1 );
			r += Volume2Linear( dst + r, src, srcSize, startSlice, sizeSlice );
		}
	} else {
		for ( auto z = start.z; z < start.z + srcSize.z; ++z ) {
			for ( auto y = start.y; y < start.y + srcSize.y; ++y ) {
				const Vec3i startLine( start.x, y, z );
				const Size3 sizeLine( size.x, 1, 1 );
				r += Volume2Linear( dst + r, src, srcSize, startLine, sizeLine );
			}
		}
	}
	return r;
}

}  // namespace

RawReader::RawReader( const std::string &fileName, const vm::Size3 &dimensions, size_t voxelSize ) :
  RawReader( fileName, dimensions, voxelSize, false )
{
}

RawReader::RawReader( const std::string &fileName,
					  const Size3 &dimensions,
					  size_t voxelSize, bool mapped ) :
  d_ptr( new RawReader__pImpl( this ) )
{
	VM_IMPL( RawReader );
	_->dataBound = { { 0, 0, 0 }, { int( dimensions.x ), int( dimensions.y ), int( dimensions.z ) } };
	//_->dimensions = dimensions;
	_->voxelSize = voxelSize;
	if ( mapped == false ) {
		_->file.open( fileName, std::ios::binary );
		if ( !_->file.is_open() ) {
			throw std::runtime_error( "RawReaderIO::failed to open file" );
		}

		_->readRegion = [this]( const Vec3i &start,
								const Size3 &size,
								unsigned char *buffer ) {
			VM_IMPL( RawReader )
			_->seekAmt = 0;
			return readRegion__( start, size, buffer );
		};

	} else {
#ifdef _WIN32
		_->mappingFile = PluginLoader::GetPluginLoader()->CreatePlugin<IMappingFile>( "windows" );
#else
		_->mappingFile = PluginLoader::GetPluginLoader()->CreatePlugin<IMappingFile>( "linux" );
#endif

		if ( _->mappingFile == nullptr ) {
			throw std::runtime_error( "Failed to load file mapping plugin" );
		}
		const auto rawBytes = dimensions.x * dimensions.y * dimensions.z * voxelSize;
		if ( _->mappingFile->Open( fileName, rawBytes, FileAccess::ReadWrite, MapAccess::ReadWrite ) == false ) {
			throw std::runtime_error( "Failed to open mapping file" );
		}
		if ( ( _->ptr = _->mappingFile->FileMemPointer( 0, rawBytes ) ) == nullptr ) {
			throw std::runtime_error( "Failed to map file" );
		}
		_->readRegion = [this]( const Vec3i &start,
								const Size3 &size,
								unsigned char *buffer ) {
			VM_IMPL( RawReader )
			return Volume2Linear( buffer, _->ptr, Size3( _->dataBound.Diagonal() ), start, size );
		};
	}
}

RawReader::~RawReader()
{
}

size_t RawReader::readRegion( const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer )
{
	VM_IMPL( RawReader )
	return _->readRegion( start, size, buffer );
}

size_t RawReader::readRegionNoBoundary( const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer )
{
	VM_IMPL( RawReader )

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
	//_->stagingBuffer.Reset();
	//auto buf = _->stagingBuffer.Alloc<unsigned char>( readSize.Prod() * _->voxelSize, false );
	
	auto buf = (unsigned char*)calloc(_->voxelSize, readSize.Prod() );
	memset( buf, 0, readSize.Prod() * _->voxelSize );

	const auto read = readRegion( { isectBound.min.x, isectBound.min.y, isectBound.min.z }, readSize, buf );

	assert( read == readSize.Prod() );

	const auto newStart = isectBound.min - readBound.min;
	
	auto ret = Linear2Volume( buffer, dstSize * _->voxelSize, newStart * _->voxelSize, buf, readSize * _->voxelSize );

	free( buf );
	return ret;
}

std::future<size_t> RawReader::asyncReadRegion( const Vec3i &start, const Vec3i &size, unsigned char *buffer, std::function<void()> cb )
{
	
	return std::async( [this]( const Vec3i &start, 
		const Vec3i &size,unsigned char *buffer, decltype( cb ) callback )
	{
		VM_IMPL( RawReader );
		const auto read = _->readRegion( start, Size3( size ), buffer );callback();
		return read;
	},start,size,buffer,std::move(cb));
}

std::future<size_t> RawReader::asyncReadRegionNoBoundary( const Vec3i &start, const Vec3i &size, unsigned char *buffer, std::function<void()> cb )
{
	return std::async( [this]( const Vec3i &start, const Vec3i &size, unsigned char *buffer, decltype( cb ) callback )
	{
		VM_IMPL( RawReader )
		const auto read = readRegionNoBoundary( start, Size3( size ), buffer );
		callback();
		return read;
	}, start, size, buffer, std::move(cb) );
}

Vec3i RawReader::GetDimension() const
{
	const auto _ = d_func();
	return _->dataBound.Diagonal();
}

size_t RawReader::GetElementSize() const
{
	const auto _ = d_func();
	return _->voxelSize;
}

std::size_t RawReader::readRegion__( const vm::Vec3i &start, const vm::Size3 &size, unsigned char *buffer )
{
	VM_IMPL( RawReader )
	assert( size.x > 0 && size.y > 0 && size.z > 0 );
	//const uint64_t startRead = ( start.x + _->dataBound.max.x * ( start.y + _->dataBound.max.y * start.z ) ) * _->voxelSize;
	const uint64_t startRead = Linear( start.ToPoint3(), Size2( _->dataBound.max.x, _->dataBound.max.y ) ) * _->voxelSize;
	if ( _->offset != startRead ) {
		_->seekAmt = startRead - _->offset;
		;
		if ( !_->file.seekg( _->seekAmt, std::ios_base::cur ) ) {
			std::cout << _->file.bad() << " " << _->file.eofbit << " " << _->file.failbit << std::endl;
			throw std::runtime_error( "RAW: Error seeking file" );
		}
		_->offset = startRead;
	}
	// Figure out how to actually read the region since it may not be a full X/Y slice and
	// we'll need to read the portions in X & Y and seek around to skip regions
	size_t read = 0;
	if ( convexRead( size ) )  // continuous read
	{
		_->file.read( reinterpret_cast<char *>( buffer ), _->voxelSize * size.x * size.y * size.z );

		read = size.x * size.y * size.z;  // voxel count

		_->offset = startRead + read * _->voxelSize;
	} else if ( size.x == _->dataBound.max.x ) {  // read by slice
		for ( auto z = start.z; z < start.z + size.z; ++z ) {
			const vm::Vec3i startSlice( start.x, start.y, z );
			const vm::Size3 sizeSlice( size.x, size.y, 1 );
			read += readRegion__( startSlice, sizeSlice, buffer + read * _->voxelSize );
		}
	} else {
		for ( auto z = start.z; z < start.z + size.z; ++z ) {
			for ( auto y = start.y; y < start.y + size.y; ++y ) {
				const vm::Vec3i startLine( start.x, y, z );
				const vm::Size3 sizeLine( size.x, 1, 1 );
				read += readRegion__( startLine, sizeLine, buffer + read * _->voxelSize );
			}
		}
	}
	return read;
}

bool RawReader::convexRead( const vm::Size3 &size ) const
{
	const auto _ = d_func();
	return ( size.x == _->dataBound.max.x && size.y == _->dataBound.max.y ) || ( size.x == _->dataBound.max.x && size.z == 1 ) || ( size.y == 1 && size.z == 1 );
}

}  // namespace vm
