# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Go project (module name: `hack`) using Go 1.23.2.

## Common Commands

### Go Development
- `go mod tidy` - Update dependencies and clean up go.mod/go.sum
- `go build` - Build the project
- `go run .` - Run the main package
- `go test ./...` - Run all tests
- `go test -v ./...` - Run tests with verbose output
- `go fmt ./...` - Format all Go code

### Development Environment
- IDE: GoLand (based on .idea directory presence)
- Platform: Windows (MINGW64)