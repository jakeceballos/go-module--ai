package model

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/ollama/ollama/api"
	"google.golang.org/genai"
)

// Test functions for ollama model conversions
func TestConvertRole(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"user role", "user", "user"},
		{"model role", "model", "assistant"},
		{"system role", "system", "system"},
		{"tool role", "tool", "tool"},
		{"unknown role passthrough", "custom", "custom"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertRole(tt.input)
			if result != tt.expected {
				t.Errorf("convertRole(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

func TestParseKeepAlive(t *testing.T) {
	tests := []struct {
		name        string
		input       string
		expected    time.Duration
		expectError bool
	}{
		{"duration string minutes", "5m", 5 * time.Minute, false},
		{"duration string seconds", "30s", 30 * time.Second, false},
		{"duration string hours", "1h", time.Hour, false},
		{"numeric seconds", "60", 60 * time.Second, false},
		{"numeric with decimal", "1.5", 1500 * time.Millisecond, false},
		{"zero", "0", 0, false},
		{"invalid format", "invalid", 0, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := parseKeepAlive(tt.input)
			if tt.expectError {
				if err == nil {
					t.Errorf("parseKeepAlive(%q) expected error, got nil", tt.input)
				}
				return
			}
			if err != nil {
				t.Errorf("parseKeepAlive(%q) unexpected error: %v", tt.input, err)
				return
			}
			if result != tt.expected {
				t.Errorf("parseKeepAlive(%q) = %v, want %v", tt.input, result, tt.expected)
			}
		})
	}
}

func TestConvertFunctionArgs(t *testing.T) {
	tests := []struct {
		name  string
		input map[string]any
	}{
		{"empty args", map[string]any{}},
		{"string arg", map[string]any{"name": "test"}},
		{"number arg", map[string]any{"count": 42}},
		{"multiple args", map[string]any{"name": "test", "count": 42, "enabled": true}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := convertFunctionArgs(tt.input)
			if err != nil {
				t.Errorf("convertFunctionArgs() unexpected error: %v", err)
				return
			}
			resultMap := result.ToMap()
			if len(resultMap) != len(tt.input) {
				t.Errorf("convertFunctionArgs() returned %d items, want %d", len(resultMap), len(tt.input))
			}
			for k, v := range tt.input {
				if resultMap[k] != v {
					t.Errorf("convertFunctionArgs()[%q] = %v, want %v", k, resultMap[k], v)
				}
			}
		})
	}
}

func TestConvertOptions(t *testing.T) {
	t.Run("empty config returns empty map", func(t *testing.T) {
		cfg := &genai.GenerateContentConfig{}
		result := convertOptions(cfg)
		if result == nil {
			t.Error("convertOptions returned nil, want empty map")
		}
		if len(result) != 0 {
			t.Errorf("convertOptions returned %d items, want 0", len(result))
		}
	})

	t.Run("temperature only", func(t *testing.T) {
		temp := float32(0.7)
		cfg := &genai.GenerateContentConfig{Temperature: &temp}
		result := convertOptions(cfg)
		if result["temperature"] != temp {
			t.Errorf("temperature = %v, want %v", result["temperature"], temp)
		}
	})

	t.Run("all options", func(t *testing.T) {
		temp := float32(0.7)
		topP := float32(0.9)
		topK := float32(40)
		seed := int32(42)
		presencePenalty := float32(0.5)
		freqPenalty := float32(0.3)

		cfg := &genai.GenerateContentConfig{
			Temperature:      &temp,
			TopP:             &topP,
			TopK:             &topK,
			MaxOutputTokens:  1024,
			StopSequences:    []string{"STOP", "END"},
			Seed:             &seed,
			PresencePenalty:  &presencePenalty,
			FrequencyPenalty: &freqPenalty,
		}

		result := convertOptions(cfg)

		if result["temperature"] != temp {
			t.Errorf("temperature = %v, want %v", result["temperature"], temp)
		}
		if result["top_p"] != topP {
			t.Errorf("top_p = %v, want %v", result["top_p"], topP)
		}
		if result["top_k"] != topK {
			t.Errorf("top_k = %v, want %v", result["top_k"], topK)
		}
		if result["num_predict"] != int32(1024) {
			t.Errorf("num_predict = %v, want %v", result["num_predict"], 1024)
		}
		if result["seed"] != seed {
			t.Errorf("seed = %v, want %v", result["seed"], seed)
		}
		if result["presence_penalty"] != presencePenalty {
			t.Errorf("presence_penalty = %v, want %v", result["presence_penalty"], presencePenalty)
		}
		if result["frequency_penalty"] != freqPenalty {
			t.Errorf("frequency_penalty = %v, want %v", result["frequency_penalty"], freqPenalty)
		}

		stops, ok := result["stop"].([]string)
		if !ok || len(stops) != 2 {
			t.Errorf("stop = %v, want [STOP END]", result["stop"])
		}
	})
}

func TestConvertContentsToMessages(t *testing.T) {
	t.Run("nil content is skipped", func(t *testing.T) {
		contents := []*genai.Content{nil}
		messages, err := convertContentsToMessages(contents)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(messages) != 0 {
			t.Errorf("expected 0 messages, got %d", len(messages))
		}
	})

	t.Run("simple text content", func(t *testing.T) {
		contents := []*genai.Content{
			{
				Role:  "user",
				Parts: []*genai.Part{{Text: "Hello"}},
			},
		}
		messages, err := convertContentsToMessages(contents)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(messages) != 1 {
			t.Errorf("expected 1 message, got %d", len(messages))
		}
		if messages[0].Role != "user" {
			t.Errorf("role = %q, want %q", messages[0].Role, "user")
		}
		if messages[0].Content != "Hello" {
			t.Errorf("content = %q, want %q", messages[0].Content, "Hello")
		}
	})

	t.Run("model role converted to assistant", func(t *testing.T) {
		contents := []*genai.Content{
			{
				Role:  "model",
				Parts: []*genai.Part{{Text: "Response"}},
			},
		}
		messages, err := convertContentsToMessages(contents)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if messages[0].Role != "assistant" {
			t.Errorf("role = %q, want %q", messages[0].Role, "assistant")
		}
	})

	t.Run("multiple text parts concatenated", func(t *testing.T) {
		contents := []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{Text: "Hello "},
					{Text: "World"},
				},
			},
		}
		messages, err := convertContentsToMessages(contents)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if messages[0].Content != "Hello World" {
			t.Errorf("content = %q, want %q", messages[0].Content, "Hello World")
		}
	})

	t.Run("inline image data", func(t *testing.T) {
		imageData := []byte{0x89, 0x50, 0x4E, 0x47} // PNG header bytes
		contents := []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					{
						InlineData: &genai.Blob{
							MIMEType: "image/png",
							Data:     imageData,
						},
					},
				},
			},
		}
		messages, err := convertContentsToMessages(contents)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(messages[0].Images) != 1 {
			t.Errorf("expected 1 image, got %d", len(messages[0].Images))
		}
	})

	t.Run("function call", func(t *testing.T) {
		contents := []*genai.Content{
			{
				Role: "model",
				Parts: []*genai.Part{
					{
						FunctionCall: &genai.FunctionCall{
							Name: "get_weather",
							Args: map[string]any{"city": "London"},
						},
					},
				},
			},
		}
		messages, err := convertContentsToMessages(contents)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if len(messages[0].ToolCalls) != 1 {
			t.Errorf("expected 1 tool call, got %d", len(messages[0].ToolCalls))
		}
		if messages[0].ToolCalls[0].Function.Name != "get_weather" {
			t.Errorf("function name = %q, want %q", messages[0].ToolCalls[0].Function.Name, "get_weather")
		}
	})

	t.Run("function response", func(t *testing.T) {
		contents := []*genai.Content{
			{
				Role: "tool",
				Parts: []*genai.Part{
					{
						FunctionResponse: &genai.FunctionResponse{
							Name:     "get_weather",
							Response: map[string]any{"temp": 20, "unit": "celsius"},
						},
					},
				},
			},
		}
		messages, err := convertContentsToMessages(contents)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if messages[0].Role != "tool" {
			t.Errorf("role = %q, want %q", messages[0].Role, "tool")
		}
		if messages[0].ToolName != "get_weather" {
			t.Errorf("tool name = %q, want %q", messages[0].ToolName, "get_weather")
		}
	})

	t.Run("nil part is skipped", func(t *testing.T) {
		contents := []*genai.Content{
			{
				Role:  "user",
				Parts: []*genai.Part{nil, {Text: "Hello"}},
			},
		}
		messages, err := convertContentsToMessages(contents)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if messages[0].Content != "Hello" {
			t.Errorf("content = %q, want %q", messages[0].Content, "Hello")
		}
	})
}

func TestConvertTools(t *testing.T) {
	t.Run("nil tool is skipped", func(t *testing.T) {
		tools := []*genai.Tool{nil}
		result := convertTools(tools)
		if len(result) != 0 {
			t.Errorf("expected 0 tools, got %d", len(result))
		}
	})

	t.Run("nil function declaration is skipped", func(t *testing.T) {
		tools := []*genai.Tool{
			{FunctionDeclarations: []*genai.FunctionDeclaration{nil}},
		}
		result := convertTools(tools)
		if len(result) != 0 {
			t.Errorf("expected 0 tools, got %d", len(result))
		}
	})

	t.Run("basic function declaration", func(t *testing.T) {
		tools := []*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:        "get_weather",
						Description: "Get the weather for a city",
					},
				},
			},
		}
		result := convertTools(tools)
		if len(result) != 1 {
			t.Errorf("expected 1 tool, got %d", len(result))
		}
		if result[0].Function.Name != "get_weather" {
			t.Errorf("name = %q, want %q", result[0].Function.Name, "get_weather")
		}
		if result[0].Function.Description != "Get the weather for a city" {
			t.Errorf("description = %q, want %q", result[0].Function.Description, "Get the weather for a city")
		}
		if result[0].Type != "function" {
			t.Errorf("type = %q, want %q", result[0].Type, "function")
		}
	})

	t.Run("function with parameters schema", func(t *testing.T) {
		tools := []*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:        "get_weather",
						Description: "Get the weather",
						Parameters: &genai.Schema{
							Type:     genai.TypeObject,
							Required: []string{"city"},
							Properties: map[string]*genai.Schema{
								"city": {
									Type:        genai.TypeString,
									Description: "City name",
								},
							},
						},
					},
				},
			},
		}
		result := convertTools(tools)
		if len(result) != 1 {
			t.Errorf("expected 1 tool, got %d", len(result))
		}
		if result[0].Function.Parameters.Type != "OBJECT" {
			t.Errorf("params type = %q, want %q", result[0].Function.Parameters.Type, "OBJECT")
		}
	})

	t.Run("function with JSON schema", func(t *testing.T) {
		tools := []*genai.Tool{
			{
				FunctionDeclarations: []*genai.FunctionDeclaration{
					{
						Name:        "get_weather",
						Description: "Get the weather",
						ParametersJsonSchema: map[string]any{
							"type":     "object",
							"required": []any{"city"},
							"properties": map[string]any{
								"city": map[string]any{
									"type":        "string",
									"description": "City name",
								},
							},
						},
					},
				},
			},
		}
		result := convertTools(tools)
		if len(result) != 1 {
			t.Errorf("expected 1 tool, got %d", len(result))
		}
		if result[0].Function.Parameters.Type != "object" {
			t.Errorf("params type = %q, want %q", result[0].Function.Parameters.Type, "object")
		}
	})
}

func TestConvertSchemaToParams(t *testing.T) {
	t.Run("simple object schema", func(t *testing.T) {
		schema := &genai.Schema{
			Type:     genai.TypeObject,
			Required: []string{"name"},
			Properties: map[string]*genai.Schema{
				"name": {Type: genai.TypeString, Description: "The name"},
			},
		}
		result := convertSchemaToParams(schema)
		if result.Type != "OBJECT" {
			t.Errorf("type = %q, want %q", result.Type, "OBJECT")
		}
		if len(result.Required) != 1 || result.Required[0] != "name" {
			t.Errorf("required = %v, want [name]", result.Required)
		}
	})

	t.Run("array schema with items", func(t *testing.T) {
		schema := &genai.Schema{
			Type: genai.TypeArray,
			Items: &genai.Schema{
				Type: genai.TypeString,
			},
		}
		result := convertSchemaToParams(schema)
		if result.Type != "ARRAY" {
			t.Errorf("type = %q, want %q", result.Type, "ARRAY")
		}
		// Items is ToolProperty, check via type assertion
		items, ok := result.Items.(api.ToolProperty)
		if !ok {
			t.Errorf("items is not ToolProperty, got %T", result.Items)
		} else if len(items.Type) == 0 || items.Type[0] != "STRING" {
			t.Errorf("items type = %v, want STRING", items.Type)
		}
	})
}

func TestConvertSchemaToProperty(t *testing.T) {
	t.Run("nil schema returns empty property", func(t *testing.T) {
		result := convertSchemaToProperty(nil)
		if len(result.Type) != 0 {
			t.Errorf("expected empty type, got %v", result.Type)
		}
	})

	t.Run("string property with description", func(t *testing.T) {
		schema := &genai.Schema{
			Type:        genai.TypeString,
			Description: "A string value",
		}
		result := convertSchemaToProperty(schema)
		if len(result.Type) == 0 || result.Type[0] != "STRING" {
			t.Errorf("type = %v, want STRING", result.Type)
		}
		if result.Description != "A string value" {
			t.Errorf("description = %q, want %q", result.Description, "A string value")
		}
	})

	t.Run("enum property", func(t *testing.T) {
		schema := &genai.Schema{
			Type: genai.TypeString,
			Enum: []string{"red", "green", "blue"},
		}
		result := convertSchemaToProperty(schema)
		if len(result.Enum) != 3 {
			t.Errorf("enum length = %d, want 3", len(result.Enum))
		}
	})

	t.Run("nested object property", func(t *testing.T) {
		schema := &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"nested": {Type: genai.TypeString},
			},
		}
		result := convertSchemaToProperty(schema)
		if result.Properties == nil {
			t.Error("expected properties to be set")
		}
	})
}

func TestConvertJsonSchemaToParams(t *testing.T) {
	t.Run("non-map schema returns default", func(t *testing.T) {
		result := convertJsonSchemaToParams("not a map")
		if result.Type != "object" {
			t.Errorf("type = %q, want %q", result.Type, "object")
		}
	})

	t.Run("full schema conversion", func(t *testing.T) {
		schema := map[string]any{
			"type":     "object",
			"required": []any{"name", "age"},
			"properties": map[string]any{
				"name": map[string]any{
					"type":        "string",
					"description": "The name",
				},
				"age": map[string]any{
					"type": "integer",
				},
			},
		}
		result := convertJsonSchemaToParams(schema)
		if result.Type != "object" {
			t.Errorf("type = %q, want %q", result.Type, "object")
		}
		if len(result.Required) != 2 {
			t.Errorf("required length = %d, want 2", len(result.Required))
		}
	})
}

func TestConvertMapToProperty(t *testing.T) {
	t.Run("basic property", func(t *testing.T) {
		m := map[string]any{
			"type":        "string",
			"description": "A string",
		}
		result := convertMapToProperty(m)
		if len(result.Type) == 0 || result.Type[0] != "string" {
			t.Errorf("type = %v, want string", result.Type)
		}
		if result.Description != "A string" {
			t.Errorf("description = %q, want %q", result.Description, "A string")
		}
	})

	t.Run("enum property", func(t *testing.T) {
		m := map[string]any{
			"type": "string",
			"enum": []any{"a", "b", "c"},
		}
		result := convertMapToProperty(m)
		if len(result.Enum) != 3 {
			t.Errorf("enum length = %d, want 3", len(result.Enum))
		}
	})

	t.Run("nested object", func(t *testing.T) {
		m := map[string]any{
			"type": "object",
			"properties": map[string]any{
				"nested": map[string]any{
					"type": "string",
				},
			},
		}
		result := convertMapToProperty(m)
		if result.Properties == nil {
			t.Error("expected properties to be set")
		}
	})

	t.Run("array with items", func(t *testing.T) {
		m := map[string]any{
			"type": "array",
			"items": map[string]any{
				"type": "string",
			},
		}
		result := convertMapToProperty(m)
		// Items is ToolProperty, check via type assertion
		items, ok := result.Items.(api.ToolProperty)
		if !ok {
			t.Errorf("items is not ToolProperty, got %T", result.Items)
		} else if len(items.Type) == 0 || items.Type[0] != "string" {
			t.Errorf("items type = %v, want string", items.Type)
		}
	})
}

func TestConvertResponse(t *testing.T) {
	t.Run("basic response", func(t *testing.T) {
		resp := &api.ChatResponse{
			Message: api.Message{
				Content: "Hello world",
			},
			Done:       true,
			DoneReason: "stop",
			Metrics: api.Metrics{
				PromptEvalCount: 10,
				EvalCount:       5,
			},
		}
		result := convertResponse(resp, true)
		if result.Content == nil {
			t.Error("expected content to be set")
		}
		if len(result.Content.Parts) == 0 {
			t.Error("expected parts to be set")
		}
		if result.Content.Parts[0].Text != "Hello world" {
			t.Errorf("text = %q, want %q", result.Content.Parts[0].Text, "Hello world")
		}
		if result.FinishReason != genai.FinishReasonStop {
			t.Errorf("finish reason = %v, want %v", result.FinishReason, genai.FinishReasonStop)
		}
		if result.TurnComplete != true {
			t.Error("expected TurnComplete to be true")
		}
	})

	t.Run("response with length finish reason", func(t *testing.T) {
		resp := &api.ChatResponse{
			Done:       true,
			DoneReason: "length",
		}
		result := convertResponse(resp, true)
		if result.FinishReason != genai.FinishReasonMaxTokens {
			t.Errorf("finish reason = %v, want %v", result.FinishReason, genai.FinishReasonMaxTokens)
		}
	})

	t.Run("response with other finish reason", func(t *testing.T) {
		resp := &api.ChatResponse{
			Done:       true,
			DoneReason: "custom_reason",
		}
		result := convertResponse(resp, true)
		if result.FinishReason != genai.FinishReasonOther {
			t.Errorf("finish reason = %v, want %v", result.FinishReason, genai.FinishReasonOther)
		}
	})

	t.Run("response with usage metadata", func(t *testing.T) {
		resp := &api.ChatResponse{
			Metrics: api.Metrics{
				PromptEvalCount: 100,
				EvalCount:       50,
			},
		}
		result := convertResponse(resp, true)
		if result.UsageMetadata == nil {
			t.Error("expected usage metadata to be set")
		}
		if result.UsageMetadata.PromptTokenCount != 100 {
			t.Errorf("prompt tokens = %d, want 100", result.UsageMetadata.PromptTokenCount)
		}
		if result.UsageMetadata.CandidatesTokenCount != 50 {
			t.Errorf("candidates tokens = %d, want 50", result.UsageMetadata.CandidatesTokenCount)
		}
		if result.UsageMetadata.TotalTokenCount != 150 {
			t.Errorf("total tokens = %d, want 150", result.UsageMetadata.TotalTokenCount)
		}
	})

	t.Run("response with timing metadata", func(t *testing.T) {
		resp := &api.ChatResponse{
			Metrics: api.Metrics{
				TotalDuration:      time.Second, // 1 second
				LoadDuration:       100 * time.Millisecond,
				PromptEvalDuration: 200 * time.Millisecond,
				EvalDuration:       700 * time.Millisecond,
			},
		}
		result := convertResponse(resp, true)
		if result.CustomMetadata == nil {
			t.Error("expected custom metadata to be set")
		}
		if result.CustomMetadata["ollama_total_duration"] != time.Second {
			t.Errorf("total duration = %v, want %v", result.CustomMetadata["ollama_total_duration"], time.Second)
		}
	})
}

func TestConvertMessageToContent(t *testing.T) {
	t.Run("text only", func(t *testing.T) {
		msg := &api.Message{Content: "Hello"}
		content, hasThinking := convertMessageToContent(msg)
		if content.Role != "model" {
			t.Errorf("role = %q, want %q", content.Role, "model")
		}
		if len(content.Parts) != 1 {
			t.Errorf("parts length = %d, want 1", len(content.Parts))
		}
		if content.Parts[0].Text != "Hello" {
			t.Errorf("text = %q, want %q", content.Parts[0].Text, "Hello")
		}
		if hasThinking {
			t.Error("expected hasThinking to be false")
		}
	})

	t.Run("with thinking content", func(t *testing.T) {
		msg := &api.Message{
			Content:  "Result",
			Thinking: "Let me think about this...",
		}
		content, hasThinking := convertMessageToContent(msg)
		if !hasThinking {
			t.Error("expected hasThinking to be true")
		}
		// Thinking should be prepended
		if len(content.Parts) != 2 {
			t.Errorf("parts length = %d, want 2", len(content.Parts))
		}
		if content.Parts[0].Text != "[Thinking] Let me think about this..." {
			t.Errorf("thinking text = %q", content.Parts[0].Text)
		}
		if content.Parts[1].Text != "Result" {
			t.Errorf("result text = %q, want %q", content.Parts[1].Text, "Result")
		}
	})

	t.Run("with tool calls", func(t *testing.T) {
		args := api.ToolCallFunctionArguments{}
		args.Set("city", "London")
		msg := &api.Message{
			ToolCalls: []api.ToolCall{
				{
					Function: api.ToolCallFunction{
						Name:      "get_weather",
						Arguments: args,
					},
				},
			},
		}
		content, _ := convertMessageToContent(msg)
		if len(content.Parts) != 1 {
			t.Errorf("parts length = %d, want 1", len(content.Parts))
		}
		if content.Parts[0].FunctionCall == nil {
			t.Error("expected function call to be set")
		}
		if content.Parts[0].FunctionCall.Name != "get_weather" {
			t.Errorf("function name = %q, want %q", content.Parts[0].FunctionCall.Name, "get_weather")
		}
	})
}

func TestSchemaToMap(t *testing.T) {
	t.Run("nil schema returns nil", func(t *testing.T) {
		result := schemaToMap(nil)
		if result != nil {
			t.Errorf("expected nil, got %v", result)
		}
	})

	t.Run("basic schema", func(t *testing.T) {
		schema := &genai.Schema{
			Type:        genai.TypeString,
			Description: "A string",
		}
		result := schemaToMap(schema)
		if result["type"] != "STRING" {
			t.Errorf("type = %v, want STRING", result["type"])
		}
		if result["description"] != "A string" {
			t.Errorf("description = %v, want %q", result["description"], "A string")
		}
	})

	t.Run("schema with enum", func(t *testing.T) {
		schema := &genai.Schema{
			Type: genai.TypeString,
			Enum: []string{"a", "b"},
		}
		result := schemaToMap(schema)
		enum, ok := result["enum"].([]string)
		if !ok || len(enum) != 2 {
			t.Errorf("enum = %v, want [a b]", result["enum"])
		}
	})

	t.Run("schema with required", func(t *testing.T) {
		schema := &genai.Schema{
			Type:     genai.TypeObject,
			Required: []string{"field1", "field2"},
		}
		result := schemaToMap(schema)
		required, ok := result["required"].([]string)
		if !ok || len(required) != 2 {
			t.Errorf("required = %v, want [field1 field2]", result["required"])
		}
	})

	t.Run("schema with properties", func(t *testing.T) {
		schema := &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"name": {Type: genai.TypeString},
			},
		}
		result := schemaToMap(schema)
		props, ok := result["properties"].(map[string]any)
		if !ok {
			t.Error("expected properties to be a map")
		}
		nameProp, ok := props["name"].(map[string]any)
		if !ok {
			t.Error("expected name property to be a map")
		}
		if nameProp["type"] != "STRING" {
			t.Errorf("name type = %v, want STRING", nameProp["type"])
		}
	})

	t.Run("schema with items", func(t *testing.T) {
		schema := &genai.Schema{
			Type:  genai.TypeArray,
			Items: &genai.Schema{Type: genai.TypeInteger},
		}
		result := schemaToMap(schema)
		items, ok := result["items"].(map[string]any)
		if !ok {
			t.Error("expected items to be a map")
		}
		if items["type"] != "INTEGER" {
			t.Errorf("items type = %v, want INTEGER", items["type"])
		}
	})
}

func TestConvertSchemaToFormat(t *testing.T) {
	t.Run("nil schema returns error", func(t *testing.T) {
		_, err := convertSchemaToFormat(nil)
		if err == nil {
			t.Error("expected error for nil schema")
		}
	})

	t.Run("valid schema", func(t *testing.T) {
		schema := &genai.Schema{
			Type: genai.TypeObject,
			Properties: map[string]*genai.Schema{
				"name": {Type: genai.TypeString},
			},
		}
		result, err := convertSchemaToFormat(schema)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		// Verify it's valid JSON
		var parsed map[string]any
		if err := json.Unmarshal(result, &parsed); err != nil {
			t.Errorf("result is not valid JSON: %v", err)
		}
		if parsed["type"] != "OBJECT" {
			t.Errorf("type = %v, want OBJECT", parsed["type"])
		}
	})
}

func TestFormatOllamaError(t *testing.T) {
	t.Run("nil error returns nil", func(t *testing.T) {
		result := formatOllamaError(nil)
		if result != nil {
			t.Errorf("expected nil, got %v", result)
		}
	})

	t.Run("status error 400", func(t *testing.T) {
		err := api.StatusError{
			StatusCode:   400,
			ErrorMessage: "bad request",
		}
		result := formatOllamaError(err)
		if result == nil {
			t.Error("expected error, got nil")
		}
		expected := "bad request (400): bad request"
		if result.Error() != expected {
			t.Errorf("error = %q, want %q", result.Error(), expected)
		}
	})

	t.Run("status error 404", func(t *testing.T) {
		err := api.StatusError{
			StatusCode:   404,
			ErrorMessage: "model not found",
		}
		result := formatOllamaError(err)
		expected := "model not found (404): model not found"
		if result.Error() != expected {
			t.Errorf("error = %q, want %q", result.Error(), expected)
		}
	})

	t.Run("status error 429", func(t *testing.T) {
		err := api.StatusError{
			StatusCode:   429,
			ErrorMessage: "too many requests",
		}
		result := formatOllamaError(err)
		expected := "rate limit exceeded (429): too many requests"
		if result.Error() != expected {
			t.Errorf("error = %q, want %q", result.Error(), expected)
		}
	})

	t.Run("status error 500", func(t *testing.T) {
		err := api.StatusError{
			StatusCode:   500,
			ErrorMessage: "internal error",
		}
		result := formatOllamaError(err)
		expected := "internal server error (500): internal error"
		if result.Error() != expected {
			t.Errorf("error = %q, want %q", result.Error(), expected)
		}
	})

	t.Run("status error 502", func(t *testing.T) {
		err := api.StatusError{
			StatusCode:   502,
			ErrorMessage: "gateway error",
		}
		result := formatOllamaError(err)
		expected := "bad gateway (502): gateway error"
		if result.Error() != expected {
			t.Errorf("error = %q, want %q", result.Error(), expected)
		}
	})

	t.Run("status error other code", func(t *testing.T) {
		err := api.StatusError{
			StatusCode:   503,
			ErrorMessage: "service unavailable",
		}
		result := formatOllamaError(err)
		expected := "ollama API error (503): service unavailable"
		if result.Error() != expected {
			t.Errorf("error = %q, want %q", result.Error(), expected)
		}
	})
}

func TestOllamaModelName(t *testing.T) {
	om := &ollamaModel{name: "llama3.2"}
	if om.Name() != "llama3.2" {
		t.Errorf("Name() = %q, want %q", om.Name(), "llama3.2")
	}
}
