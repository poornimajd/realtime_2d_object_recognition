#Poornima JD feb 2025
# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -Wextra -std=c++17 `pkg-config --cflags opencv4`

# Linker flags
LDFLAGS = `pkg-config --libs opencv4`

# Source files
SRCS = main.cpp threshold.cpp morph.cpp segment.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
TARGET = main_program

# Default target
all: $(TARGET)

# Build the executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

# Generic rule to compile all .cpp files to .o files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(OBJS) $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)
