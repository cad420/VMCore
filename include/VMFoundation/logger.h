#pragma once

#include "VMUtils/concepts.hpp"
#include <VMUtils/common.h>
#include <VMUtils/modules.hpp>
#include <VMUtils/traits.hpp>
#include <VMFoundation/foundation_config.h>
#include <functional>
#include <sstream>

using namespace std;
namespace vm
{
	enum LogLevel : uint32_t
	{
		FATAL,
		CRITICAL,
		WARNING,
		INFO,
		DEBUG,
		LOG_LEVEL_MAX = ( uint32_t )( 10000 )
	};

	struct LogContext
	{
		int line = 0;
		const char *file = nullptr;
		const char *func = nullptr;
		const char *category = nullptr;
	};

	class VMFOUNDATION_EXPORTS LogStream : NoCopy
	{
		ostringstream ss;
		LogLevel level;
		LogContext ctx;
		friend class Logger;

	public:
		LogStream( LogLevel level, LogContext ctx ) :
		  level( level ), ctx( ctx ) {}

		LogStream( LogStream &&other )noexcept
		{
			ss = std::move( other.ss );
			level = other.level;
			ctx = other.ctx;
		}
		LogStream &operator=( LogStream &&other ) noexcept
		{
			ss = std::move( other.ss );
			level = other.level;
			ctx = other.ctx;
			return *this;
		}
		template <typename T>
		LogStream &operator<<( const T &val )
		{
			ss << val;
			return *this;
		}
		~LogStream();
	};
	using LogMsgHandler = std::function<void( LogLevel, const LogContext *, const char * )>;

	class Logger__pImpl;
	class VMFOUNDATION_EXPORTS Logger : NoCopy, NoMove
	{
		VM_DECL_IMPL( Logger )
	public:
		Logger();
		Logger( const char *file, int line, const char *func );
		LogStream Log( LogLevel level );
		static LogMsgHandler InstallLogMsgHandler( LogMsgHandler handler );
		static void SetLogLevel( LogLevel level );
		static LogLevel GetLogLevel();
		static void SetLogFormat( const char *fmt );
		static void EnableCriticalAsFatal( bool enable );
		static bool IsCriticalAsFatal();
		static void EnableWarningAsFatal( bool enable );
		static bool IsWarningAsFatal();
		static void EnableRawLog( bool enable );
		~Logger();
	};


#define LOG_FATAL                                          \
	if ( vm::Logger::GetLogLevel() < vm::LogLevel::FATAL ) \
		;                                                  \
	else                                                   \
		vm::Logger( __FILE__, __LINE__, nullptr ).Log( vm::LogLevel::FATAL )

#define LOG_CRITICAL                                          \
	if ( vm::Logger::GetLogLevel() < vm::LogLevel::CRITICAL ) \
		;                                                     \
	else                                                      \
		vm::Logger( __FILE__, __LINE__, nullptr ).Log( vm::LogLevel::CRITICAL )

#define LOG_WARNING                                          \
	if ( vm::Logger::GetLogLevel() < vm::LogLevel::WARNING ) \
		;                                                    \
	else                                                     \
		vm::Logger( __FILE__, __LINE__, nullptr ).Log( vm::LogLevel::WARNING )

#define LOG_INFO                                          \
	if ( vm::Logger::GetLogLevel() < vm::LogLevel::INFO ) \
		;                                                 \
	else                                                  \
		vm::Logger( __FILE__, __LINE__, nullptr ).Log( vm::LogLevel::INFO )

#define LOG_DEBUG                                          \
	if ( vm::Logger::GetLogLevel() < vm::LogLevel::DEBUG ) \
		;                                                  \
	else                                                   \
		vm::Logger( __FILE__, __LINE__, nullptr ).Log( vm::LogLevel::DEBUG )

#define LOG_CUSTOM( CUSTOM_LEVEL )                  \
	if ( vm::Logger::GetLogLevel() < CUSTOM_LEVEL ) \
		;                                           \
	else                                            \
		vm::Logger( __FILE__, __LINE__, nullptr ).Log( vm::LogLevel( CUSTOM_LEVEL ) )

#define VM_ASSERT( expr )               \
    if (false == (expr))                  \
    {                                   \
        LOG_CRITICAL<<#expr;            \
    }
}
