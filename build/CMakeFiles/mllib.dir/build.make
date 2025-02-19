# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_SOURCE_DIR = /home/espala/Exercicios/ML-NN-IMPLEMENTATION

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/espala/Exercicios/ML-NN-IMPLEMENTATION/build

# Include any dependencies generated for this target.
include CMakeFiles/mllib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/mllib.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/mllib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mllib.dir/flags.make

CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.o: CMakeFiles/mllib.dir/flags.make
CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.o: /home/espala/Exercicios/ML-NN-IMPLEMENTATION/ML/src/LinearRegression.cpp
CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.o: CMakeFiles/mllib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/espala/Exercicios/ML-NN-IMPLEMENTATION/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.o -MF CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.o.d -o CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.o -c /home/espala/Exercicios/ML-NN-IMPLEMENTATION/ML/src/LinearRegression.cpp

CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/espala/Exercicios/ML-NN-IMPLEMENTATION/ML/src/LinearRegression.cpp > CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.i

CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/espala/Exercicios/ML-NN-IMPLEMENTATION/ML/src/LinearRegression.cpp -o CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.s

CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.o: CMakeFiles/mllib.dir/flags.make
CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.o: /home/espala/Exercicios/ML-NN-IMPLEMENTATION/ML/src/LogisticRegression.cpp
CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.o: CMakeFiles/mllib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/espala/Exercicios/ML-NN-IMPLEMENTATION/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.o -MF CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.o.d -o CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.o -c /home/espala/Exercicios/ML-NN-IMPLEMENTATION/ML/src/LogisticRegression.cpp

CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/espala/Exercicios/ML-NN-IMPLEMENTATION/ML/src/LogisticRegression.cpp > CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.i

CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/espala/Exercicios/ML-NN-IMPLEMENTATION/ML/src/LogisticRegression.cpp -o CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.s

# Object files for target mllib
mllib_OBJECTS = \
"CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.o" \
"CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.o"

# External object files for target mllib
mllib_EXTERNAL_OBJECTS =

libmllib.so: CMakeFiles/mllib.dir/ML/src/LinearRegression.cpp.o
libmllib.so: CMakeFiles/mllib.dir/ML/src/LogisticRegression.cpp.o
libmllib.so: CMakeFiles/mllib.dir/build.make
libmllib.so: CMakeFiles/mllib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/espala/Exercicios/ML-NN-IMPLEMENTATION/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library libmllib.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mllib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mllib.dir/build: libmllib.so
.PHONY : CMakeFiles/mllib.dir/build

CMakeFiles/mllib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mllib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mllib.dir/clean

CMakeFiles/mllib.dir/depend:
	cd /home/espala/Exercicios/ML-NN-IMPLEMENTATION/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/espala/Exercicios/ML-NN-IMPLEMENTATION /home/espala/Exercicios/ML-NN-IMPLEMENTATION /home/espala/Exercicios/ML-NN-IMPLEMENTATION/build /home/espala/Exercicios/ML-NN-IMPLEMENTATION/build /home/espala/Exercicios/ML-NN-IMPLEMENTATION/build/CMakeFiles/mllib.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/mllib.dir/depend

