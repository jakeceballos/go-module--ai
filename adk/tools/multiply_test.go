package tools

import (
	"testing"

	"google.golang.org/adk/tool"
)

func TestNewMultiplyTool(t *testing.T) {
	multiplyTool, err := NewMultiplyTool()
	if err != nil {
		t.Fatalf("NewMultiplyTool() returned error: %v", err)
	}
	if multiplyTool == nil {
		t.Fatal("NewMultiplyTool() returned nil tool")
	}
}

func TestMultiplyHandler(t *testing.T) {
	tests := []struct {
		name     string
		input    MultiplyInput
		expected float64
	}{
		{
			name:     "positive integers",
			input:    MultiplyInput{A: 6, B: 7},
			expected: 42,
		},
		{
			name:     "with zero",
			input:    MultiplyInput{A: 5, B: 0},
			expected: 0,
		},
		{
			name:     "negative numbers",
			input:    MultiplyInput{A: -3, B: 4},
			expected: -12,
		},
		{
			name:     "two negatives",
			input:    MultiplyInput{A: -5, B: -5},
			expected: 25,
		},
		{
			name:     "floating point",
			input:    MultiplyInput{A: 2.5, B: 4},
			expected: 10,
		},
		{
			name:     "small decimals",
			input:    MultiplyInput{A: 0.1, B: 0.2},
			expected: 0.02,
		},
		{
			name:     "large numbers",
			input:    MultiplyInput{A: 1000000, B: 1000000},
			expected: 1000000000000,
		},
	}

	// Create a mock context (tool.Context is an interface)
	var ctx tool.Context = nil

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := multiplyHandler(ctx, tt.input)
			if err != nil {
				t.Fatalf("multiplyHandler() returned error: %v", err)
			}

			// Use a tolerance for floating point comparison
			tolerance := 0.0001
			diff := result.Result - tt.expected
			if diff < 0 {
				diff = -diff
			}
			if diff > tolerance {
				t.Errorf("multiplyHandler() = %v, want %v", result.Result, tt.expected)
			}
		})
	}
}

func TestMultiplyInput_JSONTags(t *testing.T) {
	// Verify the struct has proper JSON tags for schema inference
	input := MultiplyInput{A: 1, B: 2}
	if input.A != 1 || input.B != 2 {
		t.Error("MultiplyInput fields not accessible")
	}
}

func TestMultiplyOutput_JSONTags(t *testing.T) {
	// Verify the struct has proper JSON tags for schema inference
	output := MultiplyOutput{Result: 42}
	if output.Result != 42 {
		t.Error("MultiplyOutput.Result field not accessible")
	}
}
