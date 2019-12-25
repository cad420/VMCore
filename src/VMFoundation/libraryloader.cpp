
#include <VMFoundation/libraryloader.h>
#include <iostream>
#include <filesystem>
#include <regex>

namespace vm
{


class LibraryReposity__pImpl
{
	VM_DECL_API( LibraryReposity )
public:
	LibraryReposity__pImpl( LibraryReposity *api ) :
	  q_ptr( api ) {}
	static LibraryReposity *instance;
	std::map<std::string, std::shared_ptr<Library>> repo;
};

LibraryReposity *LibraryReposity__pImpl::instance = nullptr;

LibraryReposity *LibraryReposity::GetLibraryRepo()
{
	if ( !LibraryReposity__pImpl::instance )
		LibraryReposity__pImpl::instance = new LibraryReposity;
	return LibraryReposity__pImpl::instance;
}

void LibraryReposity::AddLibrary( const std::string &name )
{
	VM_IMPL( LibraryReposity )
	if ( _->repo.find( name ) != _->repo.end() )
		return;

	std::string fullName;
#ifdef _WIN32
	fullName = name + ".dll";
#elif defined( __MACOSX__ ) || defined( __APPLE__ )
	fullName = "lib" + name + ".dylib";  // mac extension
#elof defined( __linux__ )
	fullName = "lib" + name + ".so";	 // linux extension
#endif
	try {
		auto lib = std::make_shared<Library>( fullName );
		_->repo[ name ] = lib;
	} catch ( std::exception &e ) {
		std::cerr << e.what() << std::endl;
	}
}

void LibraryReposity::AddLibraries( const std::string &directory )
{
	VM_IMPL( LibraryReposity )
	namespace fs = std::filesystem;
	for ( auto &lib : fs::directory_iterator( directory ) ) {
		const auto fullName = lib.path().filename().string();
		std::regex reg;
		std::string libName = fullName.substr( 0, fullName.find_last_of( '.' ) );
#ifdef _WIN32
		reg = std::regex( R"(.+\.dll$)" );
		if ( std::regex_match( fullName, reg ) == false ) continue;
#elif defined( __MACOSX__ ) || defined( __APPLE__ )
		reg = std::regex( R"(^lib.+\.dylib$)" );
		if ( std::regex_match( fullName, reg ) == false ) continue;
		libName = libName.substr( 3, fullName.find_last_of( '.' ) - 3 );
#elif defined( __linux__ )
		reg = std::regex( R"(^lib.+\.so$)" );
		if ( std::regex_match( fullName, reg ) == false ) continue;
		libName = libName.substr( 3, fullName.find_last_of( '.' ) - 3 );
#endif /*defined(__MACOSX__) || defined(__APPLE__)*/
		try {
			auto rp = std::make_shared<Library>( lib.path().string() );
			_->repo[ libName ] = rp;
		} catch ( std::exception &e ) {
			std::cerr << e.what() << std::endl;
		}
	}
}

void *LibraryReposity::GetSymbol( const std::string &name ) const
{
	const auto _ = d_func();
	void *sym = nullptr;
	for ( auto iter = _->repo.cbegin(); sym == nullptr && iter != _->repo.cend(); ++iter )
		sym = iter->second->Symbol( name );
	return sym;
}

void *LibraryReposity::GetSymbol( const std::string &libName, const std::string &symbolName ) const
{
	const auto _ = d_func();
	void *sym = nullptr;
	auto iter = _->repo.find( libName );
	if ( iter != _->repo.end() ) {
		sym = iter->second->Symbol( symbolName );
	}
	return sym;
}

void LibraryReposity::AddDefaultLibrary()
{
}

bool LibraryReposity::Exists( const std::string &name ) const
{
	const auto _ = d_func();
	return _->repo.find( name ) != _->repo.end();
}

LibraryReposity::~LibraryReposity()
{
}

const std::map<std::string, std::shared_ptr<Library>> &LibraryReposity::GetLibRepo() const
{
	const auto _ = d_func();
	return _->repo;
}

LibraryReposity::LibraryReposity() :
  d_ptr( new LibraryReposity__pImpl( this ) )
{
}

}  // namespace vm
