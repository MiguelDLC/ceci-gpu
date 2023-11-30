# Compiler
CC = nvcc

# Compiler flags
CFLAGS = -g -O3 -std=c++17 --generate-line-info --use_fast_math --extra-device-vectorization

# the build target executable:
TARGET = main

# Source files
SRCS = $(TARGET).cu

all: $(TARGET)

$(TARGET): $(SRCS) main.h Makefile
	$(CC) $(CFLAGS) -o $(TARGET) $(SRCS)

# This rule will clean up the generated files
clean:
	rm -f $(TARGET)