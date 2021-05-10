#include <gtest/gtest-death-test.h>
#include <gtest/gtest.h>
#include <VMFoundation/memorypool.h>
#include <VMFoundation/memoryallocationtracker.h>

TEST(test_memory, tracker){
  const size_t maxSize = 1024 * 1024 * 10; // 10Mb
  vm::MemoryAllocationTracker tracker(maxSize);
  for(int i= 0; i < 10;i++){
    const auto allocation = tracker.Allocate(4096,1<<16);
    std::cout<<allocation;
  }
}
