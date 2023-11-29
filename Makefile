# Compiler
CC = nvcc

# Compiler flags
CFLAGS = -g -O3 -std=c++17

# the build target executable:
TARGET = main

all: $(TARGET)

$(TARGET): $(TARGET).cu
	$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cu
