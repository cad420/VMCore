#pragma once

#include <string>
#include <VMCoreExtension/plugindef.h>
#include "ifile.h"

enum class FileAccess
{
	Read,	// = GENERIC_READ,
	Write,	//= GENERIC_WRITE,
	ReadWrite
};
enum class MapAccess
{
	ReadOnly,  // = PAGE_READONLY,
	ReadWrite  //= PAGE_READWRITE
};

class IMappingFile : public ::vm::IFile
{
public:
	virtual bool Open( const std::string &fileName, size_t fileSize, FileAccess fileFlags, MapAccess mapFlags ) = 0;
	virtual unsigned char *FileMemPointer( unsigned long long offset, std::size_t size ) = 0;
	virtual void DestroyFileMemPointer( unsigned char *addr ) = 0;
	virtual bool WriteCommit() = 0;
	virtual bool Close() = 0;
	virtual ~IMappingFile() = default;
};

DECLARE_PLUGIN_METADATA( IMappingFile, "vmcore.imappingfile" )
