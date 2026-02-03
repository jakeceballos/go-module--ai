// Package main demonstrates how to use the multiply tool with an ADK agent.
//
// This example creates a simple calculator agent that can multiply two numbers
// using the custom multiply tool. It shows the pattern for:
//   - Creating a custom tool using functiontool
//   - Registering tools with an LLM agent
//   - Running an agent session with user input
//
// Usage:
//
//	go run main.go
//
// The agent will respond to questions like "What is 6 times 7?" by using
// the multiply tool to calculate the result.
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/jakeceballos/ai-go/adk/model"
	"github.com/jakeceballos/ai-go/adk/tools"
	"google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/runner"
	"google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()

	// Create the LLM model - using Ollama in this example
	llm, err := model.NewOllamaModel(ctx, "qwen3:8b", nil)
	if err != nil {
		log.Fatalf("Failed to create model: %v", err)
	}

	// Create the multiply tool
	multiplyTool, err := tools.NewMultiplyTool()
	if err != nil {
		log.Fatalf("Failed to create multiply tool: %v", err)
	}

	// Create the calculator agent with the multiply tool
	calculatorAgent, err := llmagent.New(llmagent.Config{
		Name:        "calculator_agent",
		Model:       llm,
		Description: "A calculator agent that can multiply numbers",
		Instruction: `You are a helpful calculator assistant. When asked to multiply numbers,
use the multiply tool to calculate the result. Always show your work by stating
what numbers you are multiplying and what the result is.`,
		Tools: []tool.Tool{
			multiplyTool,
		},
	})
	if err != nil {
		log.Fatalf("Failed to create agent: %v", err)
	}

	// Create an in-memory session service
	sessionService := session.InMemoryService()

	// Create the runner
	r, err := runner.New(runner.Config{
		AppName:        "multiply-example",
		Agent:          calculatorAgent,
		SessionService: sessionService,
	})
	if err != nil {
		log.Fatalf("Failed to create runner: %v", err)
	}

	// Create a new session
	createResp, err := sessionService.Create(ctx, &session.CreateRequest{
		AppName: "multiply-example",
		UserID:  "user-1",
	})
	if err != nil {
		log.Fatalf("Failed to create session: %v", err)
	}

	// Example query - ask the agent to multiply two numbers
	query := "What is 6 times 7?"
	if len(os.Args) > 1 {
		query = os.Args[1]
	}

	fmt.Printf("User: %s\n\n", query)

	// Run the agent with the query
	userContent := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			{Text: query},
		},
	}

	// Execute the runner and collect responses
	for event, err := range r.Run(ctx, createResp.Session.UserID(), createResp.Session.ID(), userContent, agent.RunConfig{}) {
		if err != nil {
			log.Fatalf("Error during execution: %v", err)
		}

		// Print agent responses - look for model content in the event
		if event.Content != nil {
			for _, part := range event.Content.Parts {
				if part.Text != "" {
					fmt.Printf("Agent: %s\n", part.Text)
				}
			}
		}
	}
}
