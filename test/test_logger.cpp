#include <VMFoundation/logger.h>
#include <gtest/gtest-death-test.h>
#include <gtest/gtest.h>

using namespace vm;
using namespace std;

static string testcase = "Critical\nWarning\nInfo\nDebug\nLevel6\nLevel7\nLevel8\n";
string test()
{
	string res;
	vm::Logger::InstallLogMsgHandler( [ &res ]( LogLevel level, const LogContext *ctx, const char *msg ) {
		stringstream ss;
		ss << msg;
		res += ss.str();
	} );

	LOG_CRITICAL << "Critical";
	LOG_WARNING << "Warning";
	LOG_INFO << "Info";
	LOG_DEBUG << "Debug";
	LOG_CUSTOM( 6 ) << "Level6";
	LOG_CUSTOM( 7 ) << "Level7";
	LOG_CUSTOM( 8 ) << "Level8";

	Logger::InstallLogMsgHandler( nullptr );  // uninstall
	return res;
}

TEST(test_logger,format){
  Logger::EnableRawLog(false);
  std::cout<<test();
}

TEST( test_logger, basic )
{
  Logger::EnableRawLog(true);
  auto res = test();
  EXPECT_EQ( res, testcase );
}

TEST( test_logger, loglevel )
{
  Logger::EnableRawLog(true);
  Logger::SetLogLevel( LogLevel::DEBUG );
  auto res = test();
  string testcase0 = "Critical\nWarning\nInfo\nDebug\n";
  EXPECT_EQ( res, testcase0 );

  Logger::SetLogLevel( LogLevel::FATAL );
  string testcase1 = "";
  res = test();
  EXPECT_EQ( res, testcase1 );

  Logger::SetLogLevel( LogLevel::CRITICAL );
  string testcase2 = "Critical\n";
  res = test();
  EXPECT_EQ( res, testcase2 );
}

TEST( test_logger_DeathTest, asfatal )
{
  Logger::EnableRawLog(true);
  Logger::EnableCriticalAsFatal( true );
  EXPECT_DEATH( LOG_CRITICAL << "expect to death", "" );
  Logger::EnableCriticalAsFatal( false );

  Logger::EnableWarningAsFatal( true );
  LOG_CRITICAL<<"expect not to death";
  EXPECT_DEATH( LOG_WARNING << "expect to death", "" );
  Logger::EnableWarningAsFatal( false );
}
