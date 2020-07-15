#include <VMFoundation/logger.h>
#include <atomic>
#include <VMUtils/fmt.hpp>
#include <VMUtils/timer.hpp>
#include <cstdio>
#include <thread>

VM_BEGIN_MODULE( vm )
static LogLevel g_level = LogLevel::LOG_LEVEL_MAX;
static bool g_criticalAsFatal = false;
static bool g_warningAsFatal = false;
static const char *g_fmtStr = nullptr;
static bool g_rawLog = false;

static string GetMsgTypeLabelStr( LogLevel level )
{
	switch ( level ) {
	case LogLevel::FATAL:
		return "FATAL";
	case LogLevel::CRITICAL:
		return "CRITICAL";
	case LogLevel::WARNING:
		return "WARNING";
	case LogLevel::INFO:
		return "INFO";
	case LogLevel::DEBUG:
		return "DEBUG";
	default:
		return string( "CUSTOM" ) + to_string( uint32_t( level ) );
	}
}

static string FormatedLog( LogLevel level, const LogContext *ctx, const std::string &rawLog )
{
	const auto typeStr = GetMsgTypeLabelStr( level );
	const auto curTime = Timer::current();
	stringstream ss;
	ss << curTime.to( "%Y%m%d" ) << " " << curTime.to( "%T" );
	ss.flags( ios::right );
	ss << setw( 10 ) << typeStr << " " << rawLog << " " << ctx->file << ":" << ctx->line;
	return ss.str();
}

static bool AsFatal( LogLevel level )
{
	if ( ( level == LogLevel::FATAL ) ||
		 ( g_warningAsFatal && level == LogLevel::WARNING ) || ( g_criticalAsFatal && level == LogLevel::CRITICAL ) )
		return true;
	return false;
}

static LogMsgHandler msgHandler = nullptr;
static LogMsgHandler defaultMsgHandler = []( LogLevel level, const LogContext *ctx, const char *msg ) {
	fprintf( stderr, msg );
};

using namespace std;
LogStream::~LogStream()
{
	string log;
	if ( !g_rawLog )
		log = FormatedLog( level, &ctx, ss.str() );
	else {
		log = ss.str();
	}
	if ( msgHandler != nullptr ) {
		msgHandler( level, &ctx, log.c_str() );
	} else {
		defaultMsgHandler( level, &ctx, log.c_str() );
	}
	if ( AsFatal( level ) ) {
		exit( 1 );
	}
}

class Logger__pImpl
{
	VM_DECL_API( Logger )
public:
	Logger__pImpl( Logger *api ) :
	  q_ptr( api )
	{
	}
	LogContext ctx;
};

Logger::Logger( const char *file, int line, const char *func ) :
  d_ptr( new Logger__pImpl( this ) )
{
	VM_IMPL( Logger );
	_->ctx.line = line;
	_->ctx.file = file;
	_->ctx.func = func;
}
LogStream Logger::Log( LogLevel level )
{
	VM_IMPL( Logger );
	return LogStream( level, _->ctx );	// must be an rvalue so that one destructor call
}

LogMsgHandler Logger::InstallLogMsgHandler( LogMsgHandler handler )
{
	auto cp = msgHandler;
	msgHandler = handler;
	return cp;
}
void Logger::SetLogLevel( LogLevel level )
{
	g_level = level;
}
LogLevel Logger::GetLogLevel()
{
	return g_level;
}
bool Logger::IsWarningAsFatal()
{
	return g_warningAsFatal;
}
bool Logger::IsCriticalAsFatal()
{
	return g_criticalAsFatal;
}

void Logger::EnableCriticalAsFatal( bool enable )
{
	g_criticalAsFatal = enable;
}

void Logger::EnableWarningAsFatal( bool enable )
{
	g_warningAsFatal = enable;
}

void Logger::SetLogFormat( const char *fmt )
{
	g_fmtStr = fmt;
}

void Logger::EnableRawLog( bool enable )
{
	g_rawLog = enable;
}
Logger::~Logger()
{
}

VM_END_MODULE()
