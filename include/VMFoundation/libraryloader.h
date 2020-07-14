
#pragma once

#include <string>
#include <map>
#include <memory>
#include <VMFoundation/library.h>
#include <VMFoundation/foundation_config.h>
#include <VMUtils/common.h>

namespace vm
{
class LibraryReposity__pImpl;

class VMFOUNDATION_EXPORTS LibraryReposity
{
	VM_DECL_IMPL( LibraryReposity )
public:
	static LibraryReposity *GetLibraryRepo();

	/**
			 * \brief Add a library to the repository
			 * \param 
			 */
	void AddLibrary( const std::string &path );

	
	void AddLibrary(const std::string & directory,const std::string & libName);


	/**
			 * \brief Add all libraries given by \a directory
			 */
	void AddLibraries( const std::string &directory );

	/**
			 * \brief  return the function pointer by the given name
	*/
	void *GetSymbol( const std::string &libName ) const;

	void *GetSymbol( const std::string &libName, const std::string &symbolName ) const;

	/**
			 * \brief  Add the default Library to repository
			 */
	void AddDefaultLibrary();

	/**
			 * \brief Check whether the library exists
			 */
	bool Exists( const std::string &libName ) const;

	~LibraryReposity();

	const std::map<std::string, std::shared_ptr<Library>> &GetLibRepo() const;

private:
	LibraryReposity();
};

/**
 *  \brief return the library name if the \a fileName is a valid library name depended on the current platform
 *  or return an empty string
 */
std::string VMFOUNDATION_EXPORTS ValidateLibraryName(const std::string & fileName);

std::string VMFOUNDATION_EXPORTS MakeValidLibraryName(const std::string & libName);


}  // namespace vm