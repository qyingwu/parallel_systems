# Makefile for Go project

# Variables
BINARY_NAME=bst-equivalence
SRC_DIR=cmd/BST_opt.go

# Build the project
build:
	go build -o $(BINARY_NAME) $(SRC_DIR)

# Clean up build artifacts
clean:
	rm -f $(BINARY_NAME)

.PHONY: build clean

## example cmd to run the program
## ./myprogram -hash-workers=4 -data-workers=4 -input=./input/coarse.txt
## ./myprogram -comp-workers=16 -input=./input/fine.txt