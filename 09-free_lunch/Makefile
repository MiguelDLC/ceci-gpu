# Compiler
CC = nvcc

########## EXERCISE: add the compiler flags for fast math ##########
# Compiler flags
CFLAGS = -g -O3 -std=c++17 --generate-line-info
########## END OF EXERCISE ##########

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