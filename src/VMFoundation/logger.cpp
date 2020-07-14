#include <VMFoundation/logger.h>
#include <atomic>

VM_BEGIN_MODULE( vm )

static LogMsgHandler msgHandler = nullptr;
static LogMsgHandler defaultMsgHandler = nullptr;
static LogLevel g_level = LogLevel::LOG_LEVEL_MAX;

using namespace std;
LogStream::~LogStream(){
  if(msgHandler != nullptr)
  {
    msgHandler(level,ctx,ss.str());
  }else{
    defaultMsgHandler(level,ctx,ss.str());
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

Logger::Logger( const char *file, int line, const char *func ):d_ptr(new Logger__pImpl(this))
{
  VM_IMPL(Logger);
  _->ctx.line = line;
  _->ctx.file = file;
  _->ctx.func = func;
}
LogStream Logger::Log( LogLevel level )
{
  VM_IMPL(Logger);
  LogStream ls(level);
  ls.ctx = &_->ctx;
  return ls;
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
void Logger::SetLogFormat( const char *fmt )
{
}

VM_END_MODULE()
