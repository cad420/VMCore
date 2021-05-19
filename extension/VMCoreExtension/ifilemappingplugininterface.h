#pragma once

#include <string>
#include <VMCoreExtension/plugindef.h>
#include "ifile.h"

#if defined(__linux) || defined(__linux__) || defined(linux)
# define FILE_MAPPING_PLUGIN_KEY "linux"
#elif defined(__APPLE__)
# define FILE_MAPPING_PLUGIN_KEY "macOS"
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64)
# define FILE_MAPPING_PLUGIN_KEY "windows"
#endif

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
	/**
	 * @brief 
	 * 
	 * @todo This function should be refactored that can only be passed C data type and structure
	 * rather than std::string
	 * 
	 * 
	 * @param fileName 
	 * @param fileSize 
	 * @param fileFlags 
	 * @param mapFlags 
	 * @return true 
	 * @return false 
	 */
	virtual bool Open( const std::string &fileName, size_t fileSize, FileAccess fileFlags, MapAccess mapFlags ) = 0;
	virtual unsigned char *MemoryMap( unsigned long long offset, std::size_t size ) = 0;

	/**
	 * @brief Unmaps the pointer retunred by @ref FileMemoryPointer
	 * 
	 * @param addr
	 */
	virtual void MemoryUnmap( unsigned char *addr ) = 0;

	/**
	 * @brief Flushes all modification to file
	 * 
	 * @return true 
	 * @return false 
	 */
	virtual bool Flush() = 0;

	virtual bool Flush(void * ptr, size_t len, int flags) = 0;

	/**
	 * @brief Closes the file
	 * 
	 * @return true 
	 * @return false 
	 */

	virtual bool Close() = 0;

	virtual ~IMappingFile() = default;
};

DECLARE_PLUGIN_METADATA( IMappingFile, "vmcore.imappingfile" )
