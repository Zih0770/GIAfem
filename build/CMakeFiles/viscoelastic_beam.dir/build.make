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
include CMakeFiles/viscoelastic_beam.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/viscoelastic_beam.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/viscoelastic_beam.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/viscoelastic_beam.dir/flags.make

CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.o: CMakeFiles/viscoelastic_beam.dir/flags.make
CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.o: ../demos/viscoelastic_beam.cpp
CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.o: CMakeFiles/viscoelastic_beam.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/sssou/giafem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.o -MF CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.o.d -o CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.o -c /home/sssou/giafem/demos/viscoelastic_beam.cpp

CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sssou/giafem/demos/viscoelastic_beam.cpp > CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.i

CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sssou/giafem/demos/viscoelastic_beam.cpp -o CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.s

# Object files for target viscoelastic_beam
viscoelastic_beam_OBJECTS = \
"CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.o"

# External object files for target viscoelastic_beam
viscoelastic_beam_EXTERNAL_OBJECTS =

viscoelastic_beam: CMakeFiles/viscoelastic_beam.dir/demos/viscoelastic_beam.cpp.o
viscoelastic_beam: CMakeFiles/viscoelastic_beam.dir/build.make
viscoelastic_beam: libgiafem_lib.a
viscoelastic_beam: /home/sssou/local/mfem/lib/libmfem.so
viscoelastic_beam: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
viscoelastic_beam: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
viscoelastic_beam: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
viscoelastic_beam: CMakeFiles/viscoelastic_beam.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/sssou/giafem/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable viscoelastic_beam"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/viscoelastic_beam.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/viscoelastic_beam.dir/build: viscoelastic_beam
.PHONY : CMakeFiles/viscoelastic_beam.dir/build

CMakeFiles/viscoelastic_beam.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/viscoelastic_beam.dir/cmake_clean.cmake
.PHONY : CMakeFiles/viscoelastic_beam.dir/clean

CMakeFiles/viscoelastic_beam.dir/depend:
	cd /home/sssou/giafem/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sssou/giafem /home/sssou/giafem /home/sssou/giafem/build /home/sssou/giafem/build /home/sssou/giafem/build/CMakeFiles/viscoelastic_beam.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/viscoelastic_beam.dir/depend

