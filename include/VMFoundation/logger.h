#pragma once

#include "VMUtils/concepts.hpp"
#include <VMUtils/common.h>
#include <VMUtils/modules.hpp>
#include <VMUtils/traits.hpp>
#include <functional>
#include <sstream>

VM_BEGIN_MODULE( vm )

enum LogLevel:uint32_t
{
  FATAL,
  CRITICAL,
  WARNING,
  INFO,
  DEBUG,
  CUSTOM = DEBUG + 1,
  LOG_LEVEL_MAX = (uint32_t)(-1)
};

using LogLevelFlags = uint32_t;

class Logger__pImpl;

using namespace std;

struct LogContext
{
  int line = 0;
  const char * file = nullptr;
  const char * func = nullptr;
  const char * category = nullptr;
};

class LogStream
{
  ostringstream ss;
  LogLevel level;
  LogContext*ctx = nullptr;
  friend class Logger;
  public:
  LogStream(LogLevel level):level(level){}


  template<typename T>
  LogStream & operator<<(const T && val){ss<<val;return *this;}
  ~LogStream();
};


using LogMsgHandler = std::function<void(LogLevel,const LogContext *,const std::string &)>;

class Logger: NoCopy,NoMove
{
	VM_DECL_IMPL( Logger )
public:
	Logger() = default;
  Logger(const char * file,int line , const char * func);
  LogStream Log(LogLevel level);
  static LogMsgHandler InstallLogMsgHandler(LogMsgHandler handler);
  static void SetLogLevel(LogLevel level);
  static LogLevel GetLogLevel();
  static void SetLogFormat(const char * fmt);
};

VM_EXPORT
{

#define LOG_FATAL \
  if(Logger::GetLogLevel() <= LogLevel::FATAL); \
  else Logger(__FILE__,__LINE__,__FUNCTION__).Log(LogLevel::FATAL)

#define LOG_CRITICAL \
  if(Logger::GetLogLevel() <= LogLevel::CRITICAL); \
  else Logger(__FILE__,__LINE__,__FUNCTION__).Log(LogLevel::CRITICAL)

#define LOG_WARNING \
  if(Logger::GetLogLevel() <= LogLevel::WARNING); \
  else Logger(__FILE__,__LINE__,__FUNCTION__).Log(LogLevel::WARNING)

#define LOG_INFO \
  if(Logger::GetLogLevel() <= LogLevel::INFO); \
  else Logger(__FILE__,__LINE__,__FUNCTION__).Log(LogLevel::INFO)

#define LOG_DEBUG \
  if(Logger::GetLogLevel() <= LogLevel::Debug); \
  else Logger(__FILE__,__LINE__,__FUNCTION__).Log(LogLevel::Debug)

#define LOG_CUSTOM(CUSTOM_LEVEL) \
  if(Logger::GetLogLevel() <= CUSTOM_LEVEL); \
  esle Logger(__FILE__,__LINE__,__FUNCTION__).Log(CUSTOM_LEVEL)

}

VM_END_MODULE()
