# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sssou/giafem

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sssou/giafem/build

# Include any dependencies generated for this target.
include CMakeFiles/SG_PW.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/SG_PW.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/SG_PW.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/SG_PW.dir/flags.make

CMakeFiles/SG_PW.dir/SG_PW.cpp.o: CMakeFiles/SG_PW.dir/flags.make
CMakeFiles/SG_PW.dir/SG_PW.cpp.o: ../SG_PW.cpp
CMakeFiles/SG_PW.dir/SG_PW.cpp.o: CMakeFiles/SG_PW.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sssou/giafem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/SG_PW.dir/SG_PW.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/SG_PW.dir/SG_PW.cpp.o -MF CMakeFiles/SG_PW.dir/SG_PW.cpp.o.d -o CMakeFiles/SG_PW.dir/SG_PW.cpp.o -c /home/sssou/giafem/SG_PW.cpp

CMakeFiles/SG_PW.dir/SG_PW.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/SG_PW.dir/SG_PW.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sssou/giafem/SG_PW.cpp > CMakeFiles/SG_PW.dir/SG_PW.cpp.i

CMakeFiles/SG_PW.dir/SG_PW.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/SG_PW.dir/SG_PW.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sssou/giafem/SG_PW.cpp -o CMakeFiles/SG_PW.dir/SG_PW.cpp.s

# Object files for target SG_PW
SG_PW_OBJECTS = \
"CMakeFiles/SG_PW.dir/SG_PW.cpp.o"

# External object files for target SG_PW
SG_PW_EXTERNAL_OBJECTS =

SG_PW: CMakeFiles/SG_PW.dir/SG_PW.cpp.o
SG_PW: CMakeFiles/SG_PW.dir/build.make
SG_PW: libgiafem_lib.a
SG_PW: /home/sssou/local/mfem/lib/libmfem.so
SG_PW: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
SG_PW: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
SG_PW: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
SG_PW: CMakeFiles/SG_PW.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sssou/giafem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable SG_PW"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/SG_PW.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/SG_PW.dir/build: SG_PW
.PHONY : CMakeFiles/SG_PW.dir/build

CMakeFiles/SG_PW.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/SG_PW.dir/cmake_clean.cmake
.PHONY : CMakeFiles/SG_PW.dir/clean

CMakeFiles/SG_PW.dir/depend:
	cd /home/sssou/giafem/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sssou/giafem /home/sssou/giafem /home/sssou/giafem/build /home/sssou/giafem/build /home/sssou/giafem/build/CMakeFiles/SG_PW.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/SG_PW.dir/depend

