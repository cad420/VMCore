
#include <iostream>
#include <fstream>

#include <VMFoundation/libraryloader.h>
#include <VMFoundation/lvdreader.h>
#include <VMFoundation/pluginloader.h>
#include <VMFoundation/logger.h>

namespace vm
{
LVDReader::LVDReader( const std::string &fileName ) :
  validFlag( true ), lvdIO( nullptr )
{
	std::ifstream fileHandle;

	fileHandle.open( fileName, std::fstream::binary );
	if ( !fileHandle.is_open() ) {
		std::cout << "Can not open .lvd\n";
		validFlag = false;
		fileHandle.close();
		return;
	}

	unsigned char headerBuf[ LVD_HEADER_SIZE ];

	fileHandle.read( (char *)headerBuf, LVD_HEADER_SIZE );
	header.Decode( headerBuf );

	fileHandle.close();

	//uint32_t magicNumber;
	//fileHandle.read((char*)&magicNumber, sizeof(int));

	if ( header.magicNum != LVDFileMagicNumber ) {
		std::cout << " This is not a lvd file\n";
		validFlag = false;
		fileHandle.close();
		return;
	}

	//int vx, vy, vz, bx, by, bz;
	//int m_originalWidth, m_originalHeight, m_originalDepth;

	/*fileHandle.read((char*)&vx, sizeof(int));
		fileHandle.read((char*)&vy, sizeof(int));
		fileHandle.read((char*)&vz, sizeof(int));
		fileHandle.read((char*)&logBlockSize, sizeof(int));
		fileHandle.read((char*)&repeat, sizeof(int));
		fileHandle.read((char*)&m_originalWidth, sizeof(int));
		fileHandle.read((char*)&m_originalHeight, sizeof(int));
		fileHandle.read((char*)&m_originalDepth, sizeof(int));*/

	int vx = header.dataDim[ 0 ];
	int vy = header.dataDim[ 1 ];
	int vz = header.dataDim[ 2 ];
	logBlockSize = header.blockLengthInLog;
	padding = header.padding;

	const int originalWidth = header.originalDataDim[ 0 ];
	const int originalHeight = header.originalDataDim[ 1 ];
	const int originalDepth = header.originalDataDim[ 2 ];

	if ( logBlockSize != LogBlockSize5 && logBlockSize != LogBlockSize6 && logBlockSize != LogBlockSize7 ) {
		std::cout << "Unsupported block size\n";
		validFlag = false;
		return;
	}

	const size_t aBlockSize = BlockSize();

	// aBlockSize must be power of 2, e.g. 32 or 64
	const int bx = ( ( vx + aBlockSize - 1 ) & ~( aBlockSize - 1 ) ) / aBlockSize;
	const int by = ( ( vy + aBlockSize - 1 ) & ~( aBlockSize - 1 ) ) / aBlockSize;
	const int bz = ( ( vz + aBlockSize - 1 ) & ~( aBlockSize - 1 ) ) / aBlockSize;

	vSize = vm::Size3( ( vx ), ( vy ), ( vz ) );
	bSize = vm::Size3( bx, by, bz );
	oSize = vm::Size3( originalWidth, originalHeight, originalDepth );

	const std::size_t bytes = std::size_t( vx ) * vy * vz + LVD_HEADER_SIZE;
	// Load Library

	//auto repo = LibraryReposity::GetLibraryRepo();
	//assert( repo );
	//repo->AddLibrary( "ioplugin" );

	//lvdIO = Object::CreateObject<ysl::IFileMappingPluginInterface>("common.filemapio");
#ifdef _WIN32
	lvdIO = PluginLoader::GetPluginLoader()->CreatePlugin<IMappingFile>( "windows" );
#else defined( __linux__ ) || defined( __APPLE__ )
	lvdIO = PluginLoader::GetPluginLoader()->CreatePlugin<IMappingFile>( "linux" );
#endif
	if ( lvdIO == nullptr )
		throw std::runtime_error( "can not load ioplugin" );
	lvdIO->Open( fileName, bytes, FileAccess::ReadWrite, MapAccess::ReadWrite );

	lvdPtr = lvdIO->MemoryMap( 0, bytes );
	if ( !lvdPtr ) throw std::runtime_error( "LVDReader: bad mapping" );
}

LVDReader::LVDReader( const std::vector<std::string> &fileName, const std::vector<int> &lods )
{
	std::vector<int> levelOfDetails;
	if ( lods.size() == 0 ) {
		for ( int i = 0; i < fileName.size(); i++ )
			levelOfDetails.push_back( i );
	} 
}
void LVDReader::ReadBlock( char *dest, int blockId, int lod )
{
	const size_t blockCount = BlockDataCount();
	const auto d = lvdPtr + LVD_HEADER_SIZE;

	//fileHandle.seekg(blockCount * blockId + 36, std::ios::beg);
	memcpy( dest, d + blockCount * blockId, sizeof( char ) * blockCount );
	//fileHandle.read(dest, sizeof(char) * blockCount);
}

void LVDReader::WriteBlock( const char *src, int blockId, int lod )
{
	(void)lod;
	const size_t blockCount = BlockDataCount();
	const auto d = lvdPtr + LVD_HEADER_SIZE;
	memcpy(d + blockCount * blockId, src, sizeof( char ) * blockCount );
}

bool LVDReader::Flush( int blockId, int lod )
{
	assert( lvdPtr );
	(void)lod;
	const auto d = lvdPtr + LVD_HEADER_SIZE;
	const size_t blockCount = BlockDataCount();
	return lvdIO->Flush( d + blockCount * blockId, sizeof( char ) * blockCount, 0 );
}

bool LVDReader::Flush()
{
	LOG_CRITICAL << "LVDReader::Flush() -- Not implement yet";
	return false;
}

void LVDReader::Close()
{
	lvdIO = nullptr;
}

unsigned char *LVDReader::ReadBlock( int blockId, int lod )
{
	const size_t blockCount = BlockDataCount();
	const auto d = lvdPtr + LVD_HEADER_SIZE;
	return d + blockCount * blockId;
}

LVDReader::~LVDReader()
{
	//delete lvdIO;
}
}  // namespace ysl
