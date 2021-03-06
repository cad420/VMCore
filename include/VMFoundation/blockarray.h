#pragma once

#include <cstring>
#include <VMat/arithmetic.h>
#include <VMat/geometry.h>
#include <VMFoundation/dataarena.h>
namespace vm
{
template <typename T, int nLogBlockSize>
class Block2DArray
{
	T *m_data;
	const int m_nx, m_ny, m_nxBlocks;

public:
	Block2DArray( int x, int y, const T *d = nullptr ) :
	  m_nx( x ), m_ny( y ), m_nxBlocks( RoundUp( m_nx ) >> nLogBlockSize )
	{
		const auto nAlloc = RoundUp( m_nx ) * RoundUp( m_ny );
		m_data = AllocAligned<T>( nAlloc );
		for ( auto i = 0; i < nAlloc; i++ )
			new ( &m_data[ i ] ) T();
		if ( d )
			for ( auto y = 0; y < m_ny; y++ )
				for ( auto x = 0; x < m_nx; x++ )
					( *this )( x, y ) = d[ x + y * m_nx ];
	}

	constexpr size_t BlockSize() const
	{
		return 1 << nLogBlockSize;
	}

	int Width() const
	{
		return m_nx;
	}
	int Height() const
	{
		return m_ny;
	}

	/**
		 * \brief  Returns the multiple of BlockSize()
		 */
	int RoundUp( int x ) const
	{
		return ( x + BlockSize() - 1 ) & ~( BlockSize() - 1 );
	}

	int Block( int index ) const
	{
		return index >> nLogBlockSize;
	}

	int Offset( int index ) const
	{
		return index & ( BlockSize() - 1 );
	}

	T &operator()( int x, int y )
	{
		return const_cast<T &>( static_cast<const Block2DArray &>( *this )( x, y ) );
	}

	const T &operator()( int x, int y ) const
	{
		const auto xBlock = Block( x ), yBlock = Block( y );
		const auto xOffset = Offset( x ), yOffset = Offset( y );
		const auto index = ( m_nxBlocks * yBlock + xBlock ) * BlockSize() * BlockSize() + BlockSize() * yOffset + xOffset;
		return m_data[ index ];
	}

	void GetLinearArray( T *arr )
	{
		for ( auto y = 0; y < m_ny; y++ )
			for ( auto x = 0; x < m_nx; x++ )
				*arr++ = ( *this )( x, y );
	}

	~Block2DArray()
	{
		const auto nAlloc = RoundUp( m_nx ) * RoundUp( m_ny );
		if ( m_data )
			for ( auto i = 0; i < nAlloc; i++ )
				m_data[ i ].~T();
		FreeAligned( m_data );
	}
};

template <typename T, int nLogBlockSize>
class Block3DArray
{
	T *m_data;
	const int m_nx, m_ny, m_nz, m_nxBlocks, m_nyBlocks, m_nzBlocks;
	bool m_valid;
	// Delegate Constructor
	Block3DArray( int x, int y, int z ) :
	  m_data( nullptr ),
	  m_nx( x ),
	  m_ny( y ),
	  m_nz( z ),
	  m_nxBlocks( RoundUp( m_nx ) >> nLogBlockSize ),
	  m_nyBlocks( RoundUp( m_ny ) >> nLogBlockSize ),
	  m_nzBlocks( RoundUp( m_nz ) >> nLogBlockSize ),
	  m_valid( true )
	{
	}

public:
	Block3DArray( int x, int y, int z, const T *linearArray ) :
	  Block3DArray( x, y, z )
	{
		const auto nAlloc = RoundUp( m_nx ) * RoundUp( m_ny ) * RoundUp( m_nz );

		m_data = AllocAligned<T>( nAlloc );

		if ( m_data == nullptr ) {
			m_valid = false;
			return;
		}

		if ( linearArray ) {
#pragma omp parallel for
			for ( auto z = 0; z < m_nz; z++ )
				for ( auto y = 0; y < m_ny; y++ )
					for ( auto x = 0; x < m_nx; x++ )
						( *this )( x, y, z ) = linearArray[ x + y * m_nx + z * m_nx * m_ny ];
		}
	}
	constexpr size_t BlockSize() const { return 1 << nLogBlockSize; }
	bool Valid() const { return m_valid; }
	Block3DArray( const Block3DArray &array ) = delete;
	Block3DArray &operator=( const Block3DArray &array ) = delete;

	Block3DArray( Block3DArray &&array ) noexcept :
	  Block3DArray( array.m_nx, array.m_ny, array.m_nz )
	{
		m_data = array.m_data;
		array.m_data = nullptr;
	}

	Block3DArray &operator=( Block3DArray &&array ) noexcept
	{
		m_nx( array.x );
		m_ny( array.y );
		m_nz( array.z );
		m_nxBlocks( RoundUp( m_nx ) >> nLogBlockSize );
		m_nyBlocks( RoundUp( m_ny ) >> nLogBlockSize );
		m_nzBlocks( RoundUp( m_nz ) >> nLogBlockSize );
		m_data = array.m_data;
		array.m_data = nullptr;
		return *this;
	}

	const T *Data() const { return m_data; }

	T *Data() { return m_data; }

	int Width() const
	{
		return m_nx;
	}

	int Height() const
	{
		return m_ny;
	}

	int Depth() const
	{
		return m_nz;
	}

	int BlockWidth() const
	{
		return m_nxBlocks;
	}

	int BlockHeight() const
	{
		return m_nyBlocks;
	}

	int BlockDepth()const
	{
		return m_nzBlocks;
	}

    size_t BlockCount()const{
      return size_t(BlockWidth()) * BlockHeight() * BlockDepth();
    }

	/**
		 * \brief  Returns the multiple of BlockSize()
		 */
	size_t RoundUp( int x ) const
	{
		return ( x + BlockSize() - 1 ) & ~( BlockSize() - 1 );
	}

	int Block( int index ) const
	{
		return index >> nLogBlockSize;
	}

	int Offset( int index ) const
	{
		return index & ( BlockSize() - 1 );
	}

	T &operator()( int x, int y, int z )
	{
		return const_cast<T &>( static_cast<const Block3DArray &>( *this )( x, y, z ) );
	}

	const T &operator()( int x, int y, int z ) const
	{
		const auto xBlock = Block( x ), yBlock = Block( y ), zBlock = Block( z );
		const auto xOffset = Offset( x ), yOffset = Offset( y ), zOffset = Offset( z );
		const auto index = ( std::size_t( m_nyBlocks ) * m_nxBlocks * zBlock + std::size_t( m_nxBlocks ) * yBlock + xBlock ) * std::size_t( BlockSize() ) * BlockSize() * BlockSize() +
						   BlockSize() * BlockSize() * zOffset +
						   BlockSize() * yOffset + xOffset;
		return m_data[ index ];
	}

	Float SampleNormalized( const Point3f &p )
	{
		const auto pn = Point3f( p.x * m_nx - 0.5f, p.y * m_ny - 0.5, p.z * m_nz - 0.5 );
		return Sample( pn );
	}

	Float Sample( const Point3f &p )
	{
		const auto pi = Point3i( std::floor( p.x ), std::floor( p.y ), std::floor( p.z ) );
		const auto d = p - static_cast<Point3f>( pi );
		const auto d00 = Lerp( d.x, Sample( pi ), Sample( pi + Vector3i( 1, 0, 0 ) ) );
		const auto d10 = Lerp( d.x, Sample( pi + Vector3i( 0, 1, 0 ) ), Sample( pi + Vector3i( 1, 1, 0 ) ) );
		const auto d01 = Lerp( d.x, Sample( pi + Vector3i( 0, 0, 1 ) ), Sample( pi + Vector3i( 1, 0, 1 ) ) );
		const auto d11 = Lerp( d.x, Sample( pi + Vector3i( 0, 1, 1 ) ), Sample( pi + Vector3i( 1, 1, 1 ) ) );
		const auto d0 = Lerp( d.y, d00, d10 );
		const auto d1 = Lerp( d.y, d01, d11 );
		return Lerp( d.z, d0, d1 );
	}
	/**
	 * @brief  samples in a block addressed by 3d coordinate
	 * 
	 * @param xBlock 
	 * @param yBlock 
	 * @param zBlock 
	 * @param innerOffset 
	 * @return Float 
	 */

	Float Sample( int xBlock, int yBlock, int zBlock , const Point3f & innerOffset){
		const auto pi = Point3i( std::floor( innerOffset.x ), std::floor( innerOffset.y ), std::floor( innerOffset.z ) );
		const auto d = innerOffset - static_cast<Point3f>( pi );
		const auto d00 = Lerp( d.x, Sample(xBlock,yBlock,zBlock, pi ), Sample(xBlock,yBlock,zBlock, pi + Vector3i( 1, 0, 0 ) ) );
		const auto d10 = Lerp( d.x, Sample(xBlock,yBlock,zBlock,  pi + Vector3i( 0, 1, 0 ) ), Sample(xBlock,yBlock,zBlock,  pi + Vector3i( 1, 1, 0 ) ) );
		const auto d01 = Lerp( d.x, Sample(xBlock,yBlock,zBlock,  pi + Vector3i( 0, 0, 1 ) ), Sample(xBlock,yBlock,zBlock,  pi + Vector3i( 1, 0, 1 ) ) );
		const auto d11 = Lerp( d.x, Sample(xBlock,yBlock,zBlock,  pi + Vector3i( 0, 1, 1 ) ), Sample(xBlock,yBlock,zBlock,  pi + Vector3i( 1, 1, 1 ) ) );
		const auto d0 = Lerp( d.y, d00, d10 );
		const auto d1 = Lerp( d.y, d01, d11 );
		return Lerp( d.z, d0, d1 );
	}

	/**
	 * @brief samples in a block addressed by a flat block id
	 * 
	 * @param blockID 
	 * @param innerOffset 
	 * @return Float 
	 */

	Float Sample( size_t blockID, const Point3f & innerOffset){
		const auto pi = Point3i( std::floor( innerOffset.x ), std::floor( innerOffset.y ), std::floor( innerOffset.z ) );
		const auto d = innerOffset - static_cast<Point3f>( pi );
		const auto d00 = Lerp( d.x, Sample(blockID, pi ), Sample(blockID, pi + Vector3i( 1, 0, 0 ) ) );
		const auto d10 = Lerp( d.x, Sample(blockID, pi + Vector3i( 0, 1, 0 ) ), Sample(blockID, pi + Vector3i( 1, 1, 0 ) ) );
		const auto d01 = Lerp( d.x, Sample(blockID, pi + Vector3i( 0, 0, 1 ) ), Sample(blockID, pi + Vector3i( 1, 0, 1 ) ) );
		const auto d11 = Lerp( d.x, Sample(blockID, pi + Vector3i( 0, 1, 1 ) ), Sample(blockID, pi + Vector3i( 1, 1, 1 ) ) );
		const auto d0 = Lerp( d.y, d00, d10 );
		const auto d1 = Lerp( d.y, d01, d11 );
		return Lerp( d.z, d0, d1 );
		return 0.0;
	}

	/**
	 * @brief sample a grid point in a block addressed by 3d coordinate
	 * 
	 * @param xBlock 
	 * @param yBlock 
	 * @param zBlock 
	 * @param innerOffset 
	 * @return Float 
	 */
	Float Sample( int xBlock, int yBlock, int zBlock , const Point3i & innerOffset){
		assert(innerOffset.x >=0);
		assert(innerOffset.y >=0);
		assert(innerOffset.z >=0);
		auto blockData = BlockData(xBlock,yBlock,zBlock);
		auto sampleID = Linear(innerOffset,{BlockSize(),BlockSize()});
		return *(blockData + sampleID);
		return 0.0;
	}

	/**
	 * @brief samples a grid point in a block addressed by flat block id
	 * 
	 * @param blockID 
	 * @param innerOffset 
	 * @return Float 
	 */

	Float Sample( size_t blockID, const Point3i & innerOffset){
		assert(innerOffset.x >=0);
		assert(innerOffset.y >=0);
		assert(innerOffset.z >=0);
		auto blockData = BlockData(blockID);
		auto sampleID = Linear(innerOffset,{BlockSize(),BlockSize()});
		return *(blockData + sampleID);
	}

	Float Sample( const Point3i &p )
	{
		Bound3i bound( Point3i( 0, 0, 0 ), Point3i( m_nx, m_ny, m_nz ) );
		if ( !bound.InsideEx( p ) )
			return 0;
		return ( *this )( p.x, p.y, p.z );
	}

	T *BlockData( int blockIndex )
	{
		return const_cast<T *>( static_cast<const Block3DArray &>( *this ).BlockData( blockIndex ) );
	}

	const T *BlockData( int blockIndex ) const
	{
		return m_data + blockIndex * BlockSize() * BlockSize() * BlockSize();
	}

	T *BlockData( int xBlock, int yBlock, int zBlock )
	{
		return const_cast<T *>( static_cast<const Block3DArray &>( *this ).BlockData( xBlock, yBlock, zBlock ) );
	}

	const T *BlockData( int xBlock, int yBlock, int zBlock ) const
	{
		const auto blockIndex = zBlock * ( BlockWidth() * BlockHeight() ) + yBlock * BlockWidth() + zBlock;
		return BlockData( blockIndex );
	}

	void SetBlockData( int blockIndex, const T *blockData )
	{
		std::memcpy( BlockData( blockIndex ), blockData, BlockSize() * BlockSize() * BlockSize() * sizeof( T ) );
	}

	void SetBlockData( int xBlock, int yBlock, int zBlock, const T *blockData )
	{
		std::memcpy( BlockData( xBlock, yBlock, zBlock ), blockData, BlockSize() * BlockSize() * BlockSize() * sizeof( T ) );
	}

	void GetLinearArray( T *arr )
	{
		for ( auto z = 0; z < m_nz; z++ )
			for ( auto y = 0; y < m_ny; y++ )
				for ( auto x = 0; x < m_nx; x++ )
					*arr++ = ( *this )( x, y, z );
	}
	virtual ~Block3DArray()
	{
		const auto nAlloc = RoundUp( m_nx ) * RoundUp( m_ny );
		if ( m_data )
			for ( auto i = 0; i < nAlloc; i++ )
				m_data[ i ].~T();
		FreeAligned( m_data );
	}
};

}  // namespace ysl
