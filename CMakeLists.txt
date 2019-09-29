cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 14)
project(vmcore)


aux_source_directory(src/VMFoundation FOUNDATION_SRC)
aux_source_directory(src/VMGraphics GRAPHICS_SRC)


option(VMUTILS_SHARED_LIBRARY OFF)	 # static linking
add_subdirectory(external/VMUtils)

option(VMCORE_SHARED_LIBRARY "Set to ON to build using dynamic linking" ON)
if(VMCORE_SHARED_LIBRARY)
#add_definitions(-DVMCORE_SHARED_LIBRARY)
add_library(vmcore SHARED)
target_compile_definitions(vmcore PUBLIC -DVMCORE_SHARED_LIBRARY)
else()
add_library(vmcore STATIC)
endif()

#add_library(vmfoundation ${VMCORE_SHARED}) 
#add_dependencies(vmfoundation vmutils)
#target_sources(vmfoundation PRIVATE ${FOUNDATION_SRC})
#message(vmfoundation PUBLIC )
#target_link_libraries(vmfoundation PUBLIC vmutils)
#target_include_directories(vmfoundation PUBLIC "external/VMat/include" "external/VMUtils/include" "include" "external")

#install(TARGETS vmfoundation LIBRARY DESTINATION "lib" RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")
#install(DIRECTORY "include" DESTINATION "include/VMutils")

#add_library(vmgraphics ${VMCORE_SHARED})
#add_dependencies(vmgraphics vmutils)
#target_sources(vmgraphics PRIVATE ${GRAPHICS_SRC})
#target_link_libraries(vmgraphics PUBLIC vmutils)
#target_include_directories(vmgraphics PUBLIC "external/VMat/include" "external/VMUtils/include" "include" "external")


#install(TARGETS vmgraphics LIBRARY DESTINATION "lib" RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")
#install(DIRECTORY "include" DESTINATION "include/VMutils")


#add_dependencies(vmcore vmutils)
target_sources(vmcore PRIVATE ${GRAPHICS_SRC} ${FOUNDATION_SRC})
target_link_libraries(vmcore PUBLIC)
target_include_directories(vmcore PUBLIC "external/VMat/include" "external/VMUtils/include" "include" "external" "interface")


install(TARGETS vmcore LIBRARY DESTINATION "lib" RUNTIME DESTINATION "bin" ARCHIVE DESTINATION "lib")
install(DIRECTORY "include" DESTINATION "include/VMutils")
