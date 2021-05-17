
#pragma once


#include <memory>
#include <VMFoundation/blockarray.h>
#include <VMFoundation/lvdheader.h>
#include <VMFoundation/foundation_config.h>
#include <vector>
#include <VMUtils/ref.hpp>
#include <VMCoreExtension/ifilemappingplugininterface.h>


/**
 * This class is deprecated because of its ugly api, just for test
 */

namespace vm
{

class VMFOUNDATION_EXPORTS LVDReader
{
	std::string fileName;
	LVDHeader header;
	//AbstraFileMap* lvdIO;
	unsigned char *lvdPtr;
	vm::Size3 vSize;
	vm::Size3 bSize;
	vm::Size3 oSize;
	int logBlockSize;
	int padding;
	bool validFlag;
	enum
	{
		LVDFileMagicNumber = 277536
	};
	enum
	{
		LogBlockSize5 = 5,
		LogBlockSize6 = 6,
		LogBlockSize7 = 7
	};
	enum
	{
		LVDHeaderSize = 24
	};

	void InitLVDIO();
	void InitInfoByHeader(const LVDHeader & header);

public:
	explicit LVDReader( const std::string &fileName );
	LVDReader( const std::vector<std::string> &fileName, const std::vector<int> &lods = std::vector<int>{} );
	LVDReader(const std::string & fileName,int BlockSideInLog,const Vec3i& dataSize, int padding );
	bool Valid() const { return validFlag; }
	vm::Size3 Size( int lod = 0 ) const { return vSize; }
	vm::Size3 SizeByBlock( int lod = 0 ) const { return bSize; }
	int GetBlockPadding( int lod = 0 ) const { return padding; }
	int BlockSizeInLog( int lod = 0 ) const { return logBlockSize; }
	int BlockSize( int lod = 0 ) const { return 1 << BlockSizeInLog(); }
	int BlockDataCount( int lod = 0 ) const { return BlockSize() * BlockSize() * BlockSize(); }
	int BlockCount( int lod = 0 ) const { return bSize.x * bSize.y * bSize.z; }
	vm::Size3 OriginalDataSize( int lod = 0 ) const { return oSize; }
	template <typename T, int nLogBlockSize>
	std::shared_ptr<Block3DArray<T, nLogBlockSize>> ReadAll( int lod = 0 );
	void ReadBlock( char *dest, int blockId, int lod = 0 );
	void WriteBlock( const char *src, int blockId, int lod );
	bool Flush( int blockId, int lod );
	bool Flush();
	void Close();
	unsigned char *ReadBlock( int blockId, int lod = 0 );
	const LVDHeader &GetHeader() const { return header; }
	~LVDReader();

private:
	//std::shared_ptr<IFileMappingPluginInterface> lvdIO;		// the declaration must be behind the declaration of ~LVDReader()
	::vm::Ref<IMappingFile> lvdIO;
};

template <typename T, int nLogBlockSize>
std::shared_ptr<Block3DArray<T, nLogBlockSize>> LVDReader::ReadAll( int lod )
{
	const auto s = Size();
	const size_t bytes = s.x * s.y * s.z * sizeof( T );
	auto ptr = std::make_shared<vm::Block3DArray<T, nLogBlockSize>>( s.x, s.y, s.z, nullptr );
	//if (ptr)
	//{
	//	fileHandle.seekg((size_t)LVDHeaderSize, std::ios::beg);
	//	fileHandle.read(ptr->Data(), bytes);
	//}
	return ptr;
}
}  // namespace ysl
