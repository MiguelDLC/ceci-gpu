# Compiler
CC = nvcc

# Compiler flags
CFLAGS = -g -O3 -std=c++17

# the build target executable:
TARGET = main

# Source files
SRCS = $(TARGET).cu

all: $(TARGET)

$(TARGET): $(SRCS) main.h
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS)

# This rule will clean up the generated files
clean:
	rm -f $(TARGET)