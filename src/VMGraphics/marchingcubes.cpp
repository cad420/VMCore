//
// Created by 杨朔柳 on 2019/2/9.
//

#include <VMGraphics/marchingcubes.h>
#include <VMat/geometry.h>
#include <algorithm>
#include <cassert>

constexpr int MeshGenerator::m_edgeToVertex[ 12 ][ 6 ];
constexpr int MeshGenerator::m_triangleTable[ 256 ][ 16 ];

MeshGenerator::MeshGenerator( const unsigned char *d, vm::Size3 size ) :
  data( d ),
  dataSize( size ),
  dataXSpace( 1.0 ),
  dataYSpace( 1.0 ),
  dataZSpace( 1.0 ),
  root( nullptr )
{
	Preprocess();
}

MeshGenerator::MeshGenerator( const unsigned char *d, vm::Size3 size, vm::Vec3f space ) :
  data( d ), dataSize( size ), dataXSpace( space.x ), dataYSpace( space.y ), dataZSpace( space.z )
{
	Preprocess();
}

MeshGenerator::MeshGenerator( MeshGenerator &&generator ) noexcept
{
	root = generator.root;
	generator.root = nullptr;
	dataSize = generator.dataSize;
	data = generator.data;
	generator.data = nullptr;

	dataXSpace = generator.dataXSpace;
	dataYSpace = generator.dataYSpace;
	dataZSpace = generator.dataZSpace;

	gradient = std::move( generator.gradient );
}

MeshGenerator &MeshGenerator::operator=( MeshGenerator &&generator ) noexcept
{
	root = generator.root;
	generator.root = nullptr;
	dataSize = generator.dataSize;
	data = generator.data;
	generator.data = nullptr;

	dataXSpace = generator.dataXSpace;
	dataYSpace = generator.dataYSpace;
	dataZSpace = generator.dataZSpace;

	gradient = std::move( generator.gradient );

	return *this;
}
//std::pair<std::vector<ysl::Point3f>, std::vector<ysl::Vector3f>> MeshGenerator::GenerateMeshEx(int value)const
//{
//	std::vector<ysl::Point3f> points;
//	std::vector<ysl::Vector3f> normals;
//	TraverseOctree(root, points, normals, value, dataSize, data, gradient);
//	return { points,normals };
//}

std::shared_ptr<vm::TriangleMesh> MeshGenerator::GenerateMesh( int value ) const
{
	std::vector<vm::Point3f> triangles;
	std::vector<vm::Vector3f> normals;

	constexpr int step = 1;

	for ( int z = 0; z < dataSize.z - step; z += step ) {
		for ( int y = 0; y < dataSize.y - step; y += step ) {
			for ( int x = 0; x < dataSize.x - step; x += step ) {
				//std::cout << "??\n";
				int id = 0;
				id += data[ vm::Linear( { x, y + step, z + step }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 0 ) : ( 0 );
				id += data[ vm::Linear( { x, y + step, z }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 1 ) : ( 0 );
				id += data[ vm::Linear( { x, y, z }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 2 ) : ( 0 );
				id += data[ vm::Linear( { x, y, z + step }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 3 ) : ( 0 );
				id += data[ vm::Linear( { x + step, y + step, z + step }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 4 ) : ( 0 );
				id += data[ vm::Linear( { x + step, y + step, z }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 5 ) : ( 0 );
				id += data[ vm::Linear( { x + step, y, z }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 6 ) : ( 0 );
				id += data[ vm::Linear( { x + step, y, z + step }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 7 ) : ( 0 );

				for ( int i = 0; m_triangleTable[ id ][ i ] != -1; i += 3 ) {
					//for a single triangle mesh

					vm::Point3f tri[ 3 ];

					for ( int j = 0; j < 3; j++ ) {
						int e = m_triangleTable[ id ][ i + j ];
						int v1x = x + m_edgeToVertex[ e ][ 0 ] * step;
						int v1y = y + m_edgeToVertex[ e ][ 1 ] * step;
						int v1z = z + m_edgeToVertex[ e ][ 2 ] * step;
						int v2x = x + m_edgeToVertex[ e ][ 3 ] * step;
						int v2y = y + m_edgeToVertex[ e ][ 4 ] * step;
						int v2z = z + m_edgeToVertex[ e ][ 5 ] * step;

						int v1value = data[ vm::Linear( { v1x, v1y, v1z }, { dataSize.x, dataSize.y } ) ];
						int v2value = data[ vm::Linear( { v2x, v2y, v2z }, { dataSize.x, dataSize.y } ) ];

						//ysl::Log("%d %d", v1z,v2z);

						tri[ j ] = vm::Lerp( (float)( value - v1value ) / (float)( v2value - v1value ),
											  vm::Point3f( v1x, v1y, v1z ),
											  vm::Point3f( v2x, v2y, v2z ) );
						triangles.push_back( tri[ j ] );
						normals.push_back( -vm::Vector3f( ( gradient[ vm::Linear( { v1x, v1y, v1z }, { dataSize.x, dataSize.y } ) ] + gradient[ vm::Linear( { v2x, v2y, v2z }, { dataSize.x, dataSize.y } ) ] ) / 2 ) );
					}
					//normals.push_back(-ysl::Vector3f::Cross(tri[1]-tri[0],tri[2]-tri[1]));
				}
			}
		}
	}

	std::vector<int> indices;
	for ( int i = 0; i < triangles.size(); i++ )
		indices.push_back( i );

	//std::cout << triangles.size() << std::endl;

	return std::make_shared<vm::TriangleMesh>( vm::Transform{},
												triangles.data(),
												normals.data(),
												nullptr,
												triangles.size(),
												indices.data(),
												triangles.size() / 3 );
}

void MeshGenerator::Preprocess()
{
	// Calculate Gradient for normals

	auto size = dataSize.z * dataSize.y * dataSize.x;
	gradient.resize( size );
	for ( int z = 0; z < dataSize.z; z++ ) {
		for ( int y = 0; y < dataSize.y; y++ ) {
			for ( int x = 0; x < dataSize.x; x++ ) {
				int step;
				int x2 = x + 1, x1 = x - 1;
				int y2 = y + 1, y1 = y - 1;
				int z2 = z + 1, z1 = z - 1;
				int fx1, fx2, fy1, fy2, fz1, fz2;
				float stepx = 2, stepy = 2, stepz = 2;
				if ( x2 >= dataSize.x ) {
					fx2 = 0;
					stepx = 1;
				} else {
					fx2 = data[ vm::Linear( { x2, y, z }, { dataSize.x, dataSize.y } ) ];
				}
				if ( x1 < 0 ) {
					fx1 = 0;
					stepx = 1;
				} else {
					fx1 = data[ vm::Linear( { x1, y, z }, { dataSize.x, dataSize.y } ) ];
				}
				if ( y2 >= dataSize.y ) {
					fy2 = 0;
					stepy = 1;
				} else {
					fy2 = data[ vm::Linear( { x, y2, z }, { dataSize.x, dataSize.y } ) ];
				}
				if ( y1 < 0 ) {
					fy1 = 0;
					stepy = 1;
				} else {
					fy1 = data[ vm::Linear( { x, y1, z }, { dataSize.x, dataSize.y } ) ];
				}
				if ( z2 >= dataSize.z ) {
					fz2 = 0;
					stepz = 1;
				} else {
					fz2 = data[ vm::Linear( { x, y, z2 }, { dataSize.x, dataSize.y } ) ];
				}
				if ( z1 < 0 ) {
					fz1 = 0;
					stepz = 1;
				} else {
					fz1 = data[ vm::Linear( { x, y, z1 }, { dataSize.x, dataSize.y } ) ];
				}
				gradient[ vm::Linear( { x, y, z }, { dataSize.x, dataSize.y } ) ] = vm::Vector3f( -( fx2 - fx1 ) / stepx, -( fy2 - fy1 ) / stepy, -( fz2 - fz1 ) / stepz );
			}
		}
	}
	// Build Octree Accelerator
	//root = CreateOctree(data, ysl::Bound3i{ {0,0,0},{int(dataSize.x),int(dataSize.y),int(dataSize.z)} }, 27);
}

//ysl::Vector3f
//MarchingCubes::interpulation(int x1, int y1, int z1, int x2, int y2, int z2, int value1, int value2, int iso) {

//    return ysl::Vector3f(dataXSpace*(static_cast<float>(iso - value1)*(x2 - x1) / static_cast<float>(value1 - value2) + x1),
//                     dataYSpace*(static_cast<float>(iso - value1)*(y2 - y1) / float(value1 - value2) + y1),
//                     dataZSpace*(static_cast<float>(iso - value1)*(z2 - z1) / float(value1 - value2) + z1));
//}

MeshGenerator::OctreeNode *MeshGenerator::BuildOctree( const vm::Size3 &size, const unsigned char *d,
													   const vm::Bound3i &octreeBound,
													   const vm::Bound3i &dataBound, int thresholdVolume )
{
	//auto dx = (root->max_point.x - root->min_point.x);
	//auto dy = (root->max_point.y - root->min_point.y);
	//auto dz = (root->max_point.z - root->min_point.z);
	auto newNode = new OctreeNode( octreeBound, dataBound );
	// Leaf Node
	if ( dataBound.SurfaceArea() < thresholdVolume ) {
		const auto diagnal = dataBound.Diagonal();
		auto minV = std::numeric_limits<unsigned char>::max();
		auto maxV = std::numeric_limits<unsigned char>::min();

		for ( int z = 0; z < diagnal.z; z++ ) {
			for ( int y = 0; y < diagnal.y; y++ ) {
				for ( int x = 0; x < diagnal.x; x++ ) {
					const auto global = dataBound.min + vm::Vector3i{ x, y, z };
					const auto i = global.x + size.x * ( global.y + global.z * size.y );
					const unsigned char dd = d[ i ];
					minV = ( std::min )( dd, minV );
					maxV = ( std::max )( dd, maxV );
				}
			}
		}

		newNode->minValue = minV;
		newNode->maxValue = maxV;

		//std::cout << minV << " " << maxV << std::endl;

		return newNode;
	}

	const auto octreeBoundDiagnal = octreeBound.Diagonal();
	assert( octreeBoundDiagnal.x % 2 == 0 );
	assert( octreeBoundDiagnal.y % 2 == 0 );
	assert( octreeBoundDiagnal.z % 2 == 0 );

	const auto semiOctreeBoundDiagnal = octreeBoundDiagnal / 2;

	auto dx1 = semiOctreeBoundDiagnal.x;
	auto dy1 = semiOctreeBoundDiagnal.y;
	auto dz1 = semiOctreeBoundDiagnal.z;
	auto dx2 = dx1;
	auto dy2 = dy1;
	auto dz2 = dz1;

	//size of octree
	vm::Point3i offset[ 8 ][ 2 ] = { { { 0, 0, 0 }, { dx1, dy1, dz1 } },
									  { { dx1, 0, 0 }, { dx1 + dx2, dy1, dz1 } },
									  { { 0, dy1, 0 }, { dx1, dy1 + dy2, dz1 } },
									  { { dx1, dy1, 0 }, { dx1 + dx2, dy1 + dy2, dz1 } },
									  { { 0, 0, dz1 }, { dx1, dy1, dz1 + dz2 } },
									  { { dx1, 0, dz1 }, { dx1 + dx2, dy1, dz1 + dz2 } },
									  { { 0, dy1, dz1 }, { dx1, dy1 + dy2, dz1 + dz2 } },
									  { { dx1, dy1, dz1 }, { dx1 + dx2, dy1 + dy2, dz1 + dz2 } } };

	//size of volume data
	bool subdivide[ 8 ] = { true, true, true, true, true, true, true, true };

	const auto dataBoundDiagnal = dataBound.Diagonal();

	auto d_dx1 = std::min( dataBoundDiagnal.x, dx1 );
	auto d_dy1 = std::min( dataBoundDiagnal.y, dy1 );
	auto d_dz1 = std::min( dataBoundDiagnal.z, dz1 );

	auto d_dx2 = std::max( dataBoundDiagnal.x - d_dx1, 0 );
	auto d_dy2 = std::max( dataBoundDiagnal.y - d_dy1, 0 );
	auto d_dz2 = std::max( dataBoundDiagnal.z - d_dz1, 0 );

	if ( d_dx2 == 0 ) {
		subdivide[ 1 ] = false;
		subdivide[ 3 ] = false;
		subdivide[ 5 ] = false;
		subdivide[ 7 ] = false;
	}
	if ( d_dy2 == 0 ) {
		subdivide[ 2 ] = false;
		subdivide[ 3 ] = false;
		subdivide[ 6 ] = false;
		subdivide[ 7 ] = false;
	}
	if ( d_dz2 == 0 ) {
		subdivide[ 4 ] = false;
		subdivide[ 5 ] = false;
		subdivide[ 6 ] = false;
		subdivide[ 7 ] = false;
	}

	vm::Point3i data_offset[ 8 ][ 2 ] = { { { 0, 0, 0 }, { d_dx1, d_dy1, d_dz1 } },
										   { { d_dx1, 0, 0 }, { d_dx1 + d_dx2, d_dy1, d_dz1 } },
										   { { 0, d_dy1, 0 }, { d_dx1, d_dy1 + d_dy2, d_dz1 } },
										   { { d_dx1, d_dy1, 0 }, { d_dx1 + d_dx2, d_dy1 + d_dy2, d_dz1 } },
										   { { 0, 0, d_dz1 }, { d_dx1, d_dy1, d_dz1 + d_dz2 } },
										   { { d_dx1, 0, d_dz1 }, { d_dx1 + d_dx2, d_dy1, d_dz1 + d_dz2 } },
										   { { 0, d_dy1, d_dz1 }, { d_dx1, d_dy1 + d_dy2, d_dz1 + d_dz2 } },
										   { { d_dx1, d_dy1, d_dz1 }, { d_dx1 + d_dx2, d_dy1 + d_dy2, d_dz1 + d_dz2 } } };

	auto minV = std::numeric_limits<unsigned char>::max();
	auto maxV = std::numeric_limits<unsigned char>::min();

	for ( int z = 0; z < 2; z++ ) {
		for ( int y = 0; y < 2; y++ ) {
			for ( int x = 0; x < 2; x++ ) {
				int i = x + y * 2 + z * 4;
				if ( subdivide[ i ] ) {
					const auto ob = vm::Bound3i{ octreeBound.min + offset[ i ][ 0 ], octreeBound.min + offset[ i ][ 1 ] };
					const auto db = vm::Bound3i{ dataBound.min + data_offset[ i ][ 0 ], dataBound.min + data_offset[ i ][ 1 ] };
					const auto child = BuildOctree( size, d, ob, db, thresholdVolume );
					minV = ( std::min )( ( child->minValue ), minV );
					maxV = ( std::max )( ( child->maxValue ), maxV );
					newNode->children[ i ] = child;
					newNode->leaf = false;
				}
				//else {
				//    new_node = new octree_node(root->min_point + offset[i][0], root->min_point + offset[i][1], root->min_point + offset[i][0], root->min_point + offset[i][0], subdivide[i]);
				//}
			}
		}
	}
	newNode->minValue = minV;
	newNode->maxValue = maxV;
	return newNode;
}

void MeshGenerator::DestroyOctree( MeshGenerator::OctreeNode *octree )
{
	delete octree;
}

MeshGenerator::OctreeNode *MeshGenerator::CreateOctree( const unsigned char *d, const vm::Bound3i &bound, int thVolume )
{
	vm::Size3 dataSize( bound.max.x, bound.max.y, bound.max.z );
	vm::Bound3i octreeBound;
	octreeBound.min = vm::Point3i{ 0, 0, 0 };
	octreeBound.max = vm::Point3i{ vm::NextPowerOfTwo( (std::uint32_t)bound.max.x ), vm::NextPowerOfTwo( (std::uint32_t)bound.max.y ), vm::NextPowerOfTwo( (std::uint32_t)bound.max.z ) };

	return BuildOctree( dataSize, d, octreeBound, bound, thVolume );
	//return new OctreeNode(octreeBound, bound, thVolume);
}

void MeshGenerator::TraverseOctree( MeshGenerator::OctreeNode *root, std::vector<vm::Point3f> &triangles,
									std::vector<vm::Vector3f> &normals, int value, const vm::Size3 &dataSize,
									const unsigned char *data, const std::vector<vm::Vector3f> &gradient )
{
	if ( !root ) return;
	assert( root->minValue <= root->maxValue );
	if ( root->minValue > value || root->maxValue < value )
		return;
	if ( root->leaf == true ) {
		const auto min = root->dataBound.min;
		const auto diagnal = root->dataBound.Diagonal();

		// MarchingCubes
		for ( int z = 0; z < diagnal.z - 1; z++ ) {
			for ( int y = 0; y < diagnal.y - 1; y++ ) {
				for ( int x = 0; x < diagnal.x - 1; x++ ) {
					int id = 0;
					id += data[ vm::Linear( min + vm::Vector3i{ x, y + 1, z + 1 }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 0 ) : ( 0 );
					id += data[ vm::Linear( min + vm::Vector3i{ x, y + 1, z }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 1 ) : ( 0 );
					id += data[ vm::Linear( min + vm::Vector3i{ x, y, z }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 2 ) : ( 0 );
					id += data[ vm::Linear( min + vm::Vector3i{ x, y, z + 1 }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 3 ) : ( 0 );
					id += data[ vm::Linear( min + vm::Vector3i{ x + 1, y + 1, z + 1 }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 4 ) : ( 0 );
					id += data[ vm::Linear( min + vm::Vector3i{ x + 1, y + 1, z }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 5 ) : ( 0 );
					id += data[ vm::Linear( min + vm::Vector3i{ x + 1, y, z }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 6 ) : ( 0 );
					id += data[ vm::Linear( min + vm::Vector3i{ x + 1, y, z + 1 }, { dataSize.x, dataSize.y } ) ] > value ? ( 1 << 7 ) : ( 0 );

					for ( int i = 0; m_triangleTable[ id ][ i ] != -1; i += 3 ) {
						//for a single triangle mesh
						for ( int j = 0; j < 3; j++ ) {
							int e = m_triangleTable[ id ][ i + j ];
							int v1x = x + m_edgeToVertex[ e ][ 0 ];
							int v1y = y + m_edgeToVertex[ e ][ 1 ];
							int v1z = z + m_edgeToVertex[ e ][ 2 ];
							int v2x = x + m_edgeToVertex[ e ][ 3 ];
							int v2y = y + m_edgeToVertex[ e ][ 4 ];
							int v2z = z + m_edgeToVertex[ e ][ 5 ];

							const auto p1 = vm::Point3i{ v1x, v1y, v1z };
							const auto p2 = vm::Point3i{ v2x, v2y, v2z };

							unsigned char v1value = data[ vm::Linear( p1 + min, { dataSize.x, dataSize.y } ) ];
							unsigned char v2value = data[ vm::Linear( p2 + min, { dataSize.x, dataSize.y } ) ];

							//triangles.push_back(interpulation(v1x, v1y, v1z, v2x, v2y, v2z, v1value, v2value, value));
							triangles.push_back( vm::Lerp( float( ( value - v1value ) / ( v2value - v1value ) ), p1, p2 ) );
							normals.push_back( vm::Vector3f( ( gradient[ vm::Linear( { v1x, v1y, v1z }, { dataSize.x, dataSize.y } ) ] + gradient[ v2x, v2y, v2z ] ) / 2 ) );
							//normals.push_back(gradient[toIndex(v1x,v1y,v1z)] +static_cast<float>(value-v1value)*(gradient[toIndex(v2x,v2y,v2z)]-gradient[toIndex(v1x,v1y,v1z)])/(v2value-v1value));
						}
					}
				}
			}
		}
	}
	for ( int i = 0; i < 8; i++ )
		if ( root->children[ i ] )
			TraverseOctree( root->children[ i ], triangles, normals, value, dataSize, data, gradient );
}
