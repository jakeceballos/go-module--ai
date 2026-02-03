// Package tools provides example tools for use with ADK agents.
package tools

import (
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
)

// MultiplyInput defines the input parameters for the multiply tool.
type MultiplyInput struct {
	// A is the first number to multiply.
	A float64 `json:"a"`
	// B is the second number to multiply.
	B float64 `json:"b"`
}

// MultiplyOutput defines the output of the multiply tool.
type MultiplyOutput struct {
	// Result is the product of the two input numbers.
	Result float64 `json:"result"`
}

// multiplyHandler is the function that performs the multiplication.
func multiplyHandler(ctx tool.Context, input MultiplyInput) (MultiplyOutput, error) {
	return MultiplyOutput{
		Result: input.A * input.B,
	}, nil
}

// NewMultiplyTool creates a new tool that multiplies two numbers together.
// The tool takes two numbers (a and b) as input and returns their product.
//
// Example usage:
//
//	multiplyTool, err := tools.NewMultiplyTool()
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	agent := llmagent.New(llmagent.Config{
//	    Name:  "calculator_agent",
//	    Model: model,
//	    Tools: []tool.Tool{multiplyTool},
//	})
func NewMultiplyTool() (tool.Tool, error) {
	return functiontool.New(functiontool.Config{
		Name:        "multiply",
		Description: "Multiplies two numbers together and returns the result. Use this tool when you need to calculate the product of two numbers.",
	}, multiplyHandler)
}
