# Compiler
CC = clang++

# Compiler flags
CFLAGS = -g -O3 -std=c++17 -ltbb -lpthread -lstdc++fs

# the build target executable:
TARGET = main

# Source files
SRCS = $(TARGET).cpp

all: $(TARGET)

$(TARGET): $(SRCS) main.h Makefile
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS)

# This rule will clean up the generated files
clean:
	rm -f $(TARGET)