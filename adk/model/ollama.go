package model

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"net/url"
	"strconv"
	"time"

	"github.com/ollama/ollama/api"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// ollamaModel implements the model.LLM interface for Ollama models.
type ollamaModel struct {
	client         *api.Client
	name           string
	keepAlive      string
	enableThinking string
	baseOptions    map[string]any
}

// OllamaConfig holds configuration options for creating an Ollama model.
type OllamaConfig struct {
	// Host is the Ollama server URL. If empty, uses OLLAMA_HOST env var or defaults to localhost:11434.
	Host string
	// KeepAlive controls how long the model stays loaded in memory.
	// Can be a duration string (e.g., "5m", "10s") or a number in seconds.
	// Set to "0" to unload immediately after generating a response.
	// If empty, uses Ollama's default (5 minutes).
	KeepAlive string
	// EnableThinking enables reasoning output for supported models.
	// Can be "high", "medium", "low", or empty to disable.
	EnableThinking string
	// Options provides additional Ollama-specific model options.
	// These will be merged with options derived from genai.GenerateContentConfig.
	// Common options include:
	//   - num_ctx: Context window size (e.g., 2048, 4096)
	//   - num_predict: Maximum tokens to generate (overrides MaxOutputTokens)
	//   - repeat_penalty: Penalize repetition (default: 1.1)
	//   - repeat_last_n: How far back to look for repetition (default: 64)
	//   - min_p: Minimum probability threshold
	//   - tfs_z: Tail free sampling parameter
	//   - mirostat: Mirostat sampling mode (0, 1, or 2)
	//   - mirostat_tau: Mirostat target entropy
	//   - mirostat_eta: Mirostat learning rate
	// See: https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
	Options map[string]any
}

// NewOllamaModel creates a new Ollama model that implements the model.LLM interface.
// If cfg is nil or cfg.Host is empty, it uses the OLLAMA_HOST environment variable
// or defaults to http://localhost:11434.
func NewOllamaModel(ctx context.Context, modelName string, cfg *OllamaConfig) (model.LLM, error) {
	var client *api.Client
	var err error

	if cfg != nil && cfg.Host != "" {
		baseURL, parseErr := url.Parse(cfg.Host)
		if parseErr != nil {
			return nil, fmt.Errorf("invalid host URL: %w", parseErr)
		}
		client = api.NewClient(baseURL, nil)
	} else {
		client, err = api.ClientFromEnvironment()
		if err != nil {
			return nil, fmt.Errorf("failed to create ollama client: %w", err)
		}
	}

	om := &ollamaModel{
		client: client,
		name:   modelName,
	}

	if cfg != nil {
		om.keepAlive = cfg.KeepAlive
		om.enableThinking = cfg.EnableThinking
		om.baseOptions = cfg.Options
	}

	return om, nil
}

func (o *ollamaModel) Name() string {
	return o.name
}

// GenerateContent implements the model.LLM interface.
func (o *ollamaModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	chatReq, err := o.convertRequest(req)
	if err != nil {
		return func(yield func(*model.LLMResponse, error) bool) {
			yield(nil, fmt.Errorf("failed to convert request: %w", err))
		}
	}

	if stream {
		return o.generateStream(ctx, chatReq)
	}
	return o.generate(ctx, chatReq)
}

// convertRequest converts an ADK LLMRequest to an Ollama ChatRequest.
func (o *ollamaModel) convertRequest(req *model.LLMRequest) (*api.ChatRequest, error) {
	messages, err := convertContentsToMessages(req.Contents)
	if err != nil {
		return nil, err
	}

	chatReq := &api.ChatRequest{
		Model:    o.name,
		Messages: messages,
	}

	// Set keep_alive from config
	if o.keepAlive != "" {
		duration, err := parseKeepAlive(o.keepAlive)
		if err == nil {
			chatReq.KeepAlive = &api.Duration{Duration: duration}
		}
	}

	// Convert tools if present
	if req.Config != nil && len(req.Config.Tools) > 0 {
		chatReq.Tools = convertTools(req.Config.Tools)
	}

	// Convert generation config options
	if req.Config != nil {
		chatReq.Options = convertOptions(req.Config)
	}

	// Merge base options from OllamaConfig (they override config options)
	if len(o.baseOptions) > 0 {
		if chatReq.Options == nil {
			chatReq.Options = make(map[string]any)
		}
		for k, v := range o.baseOptions {
			chatReq.Options[k] = v
		}
	}

	if req.Config != nil {

		// Handle response format (JSON output)
		if req.Config.ResponseMIMEType == "application/json" {
			if req.Config.ResponseSchema != nil {
				// Use structured schema
				if schemaJSON, err := convertSchemaToFormat(req.Config.ResponseSchema); err == nil {
					chatReq.Format = schemaJSON
				}
			} else if req.Config.ResponseJsonSchema != nil {
				// Use raw JSON schema
				if jsonBytes, err := json.Marshal(req.Config.ResponseJsonSchema); err == nil {
					chatReq.Format = jsonBytes
				}
			} else {
				// Just request JSON format
				chatReq.Format = json.RawMessage(`"json"`)
			}
		}

		// Handle logprobs
		if req.Config.ResponseLogprobs {
			chatReq.Logprobs = true
			if req.Config.Logprobs != nil && *req.Config.Logprobs > 0 {
				chatReq.TopLogprobs = int(*req.Config.Logprobs)
			}
		}

		// Handle system instruction
		if req.Config.SystemInstruction != nil && len(req.Config.SystemInstruction.Parts) > 0 {
			var systemContent string
			for _, part := range req.Config.SystemInstruction.Parts {
				if part.Text != "" {
					systemContent += part.Text
				}
			}
			if systemContent != "" {
				// Prepend system message
				systemMsg := api.Message{
					Role:    "system",
					Content: systemContent,
				}
				chatReq.Messages = append([]api.Message{systemMsg}, chatReq.Messages...)
			}
		}
	}

	// Set thinking mode if configured
	if o.enableThinking != "" {
		thinkVal := &api.ThinkValue{Value: o.enableThinking}
		chatReq.Think = thinkVal
	}

	return chatReq, nil
}

// convertContentsToMessages converts genai.Content slice to Ollama messages.
func convertContentsToMessages(contents []*genai.Content) ([]api.Message, error) {
	var messages []api.Message

	for _, content := range contents {
		if content == nil {
			continue
		}

		msg := api.Message{
			Role: convertRole(content.Role),
		}

		for _, part := range content.Parts {
			if part == nil {
				continue
			}

			switch {
			case part.Text != "":
				msg.Content += part.Text

			case part.InlineData != nil:
				// Handle image data
				msg.Images = append(msg.Images, api.ImageData(part.InlineData.Data))

			case part.FunctionCall != nil:
				// Convert function call to tool call
				args, err := convertFunctionArgs(part.FunctionCall.Args)
				if err != nil {
					return nil, fmt.Errorf("failed to convert function args: %w", err)
				}
				msg.ToolCalls = append(msg.ToolCalls, api.ToolCall{
					Function: api.ToolCallFunction{
						Name:      part.FunctionCall.Name,
						Arguments: args,
					},
				})

			case part.FunctionResponse != nil:
				// Function response - set as tool message
				msg.Role = "tool"
				msg.ToolName = part.FunctionResponse.Name
				if part.FunctionResponse.Response != nil {
					respBytes, err := json.Marshal(part.FunctionResponse.Response)
					if err != nil {
						return nil, fmt.Errorf("failed to marshal function response: %w", err)
					}
					msg.Content = string(respBytes)
				}
			}
		}

		messages = append(messages, msg)
	}

	return messages, nil
}

// convertRole maps genai roles to Ollama roles.
func convertRole(role string) string {
	switch role {
	case "user":
		return "user"
	case "model":
		return "assistant"
	case "system":
		return "system"
	case "tool":
		return "tool"
	default:
		return role
	}
}

// convertFunctionArgs converts map[string]any to ToolCallFunctionArguments.
func convertFunctionArgs(args map[string]any) (api.ToolCallFunctionArguments, error) {
	result := api.ToolCallFunctionArguments{}
	for k, v := range args {
		result.Set(k, v)
	}
	return result, nil
}

// convertTools converts genai tools to Ollama tools.
func convertTools(tools []*genai.Tool) []api.Tool {
	var ollamaTools []api.Tool

	for _, tool := range tools {
		if tool == nil {
			continue
		}

		for _, fn := range tool.FunctionDeclarations {
			if fn == nil {
				continue
			}

			ollamaTool := api.Tool{
				Type: "function",
				Function: api.ToolFunction{
					Name:        fn.Name,
					Description: fn.Description,
				},
			}

			// Convert parameters schema
			if fn.Parameters != nil {
				ollamaTool.Function.Parameters = convertSchemaToParams(fn.Parameters)
			} else if fn.ParametersJsonSchema != nil {
				// Convert raw JSON schema to ToolFunctionParameters
				ollamaTool.Function.Parameters = convertJsonSchemaToParams(fn.ParametersJsonSchema)
			}

			ollamaTools = append(ollamaTools, ollamaTool)
		}
	}

	return ollamaTools
}

// convertSchemaToParams converts a genai.Schema to Ollama ToolFunctionParameters.
func convertSchemaToParams(schema *genai.Schema) api.ToolFunctionParameters {
	params := api.ToolFunctionParameters{
		Type:     string(schema.Type),
		Required: schema.Required,
	}

	if len(schema.Properties) > 0 {
		props := api.NewToolPropertiesMap()
		for name, prop := range schema.Properties {
			props.Set(name, convertSchemaToProperty(prop))
		}
		params.Properties = props
	}

	if schema.Items != nil {
		params.Items = convertSchemaToProperty(schema.Items)
	}

	return params
}

// convertSchemaToProperty converts a genai.Schema to an Ollama ToolProperty.
func convertSchemaToProperty(schema *genai.Schema) api.ToolProperty {
	if schema == nil {
		return api.ToolProperty{}
	}

	prop := api.ToolProperty{
		Type:        api.PropertyType{string(schema.Type)},
		Description: schema.Description,
	}

	if len(schema.Enum) > 0 {
		prop.Enum = make([]any, len(schema.Enum))
		for i, e := range schema.Enum {
			prop.Enum[i] = e
		}
	}

	if len(schema.Properties) > 0 {
		props := api.NewToolPropertiesMap()
		for name, p := range schema.Properties {
			props.Set(name, convertSchemaToProperty(p))
		}
		prop.Properties = props
	}

	if schema.Items != nil {
		prop.Items = convertSchemaToProperty(schema.Items)
	}

	return prop
}

// convertJsonSchemaToParams converts a raw JSON schema (map[string]any) to ToolFunctionParameters.
func convertJsonSchemaToParams(schema any) api.ToolFunctionParameters {
	params := api.ToolFunctionParameters{
		Type: "object",
	}

	schemaMap, ok := schema.(map[string]any)
	if !ok {
		return params
	}

	if t, ok := schemaMap["type"].(string); ok {
		params.Type = t
	}

	if required, ok := schemaMap["required"].([]any); ok {
		for _, r := range required {
			if s, ok := r.(string); ok {
				params.Required = append(params.Required, s)
			}
		}
	}

	if props, ok := schemaMap["properties"].(map[string]any); ok {
		propsMap := api.NewToolPropertiesMap()
		for name, propVal := range props {
			if propMap, ok := propVal.(map[string]any); ok {
				propsMap.Set(name, convertMapToProperty(propMap))
			}
		}
		params.Properties = propsMap
	}

	return params
}

// convertMapToProperty converts a map[string]any to an Ollama ToolProperty.
func convertMapToProperty(m map[string]any) api.ToolProperty {
	prop := api.ToolProperty{}

	if t, ok := m["type"].(string); ok {
		prop.Type = api.PropertyType{t}
	}
	if desc, ok := m["description"].(string); ok {
		prop.Description = desc
	}
	if enum, ok := m["enum"].([]any); ok {
		prop.Enum = enum
	}

	if props, ok := m["properties"].(map[string]any); ok {
		propsMap := api.NewToolPropertiesMap()
		for name, propVal := range props {
			if propMap, ok := propVal.(map[string]any); ok {
				propsMap.Set(name, convertMapToProperty(propMap))
			}
		}
		prop.Properties = propsMap
	}

	if items, ok := m["items"].(map[string]any); ok {
		itemProp := convertMapToProperty(items)
		prop.Items = itemProp
	}

	return prop
}

// convertOptions converts genai config to Ollama options.
// See https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
func convertOptions(cfg *genai.GenerateContentConfig) map[string]any {
	opts := map[string]any{}

	// Standard parameters from genai.GenerateContentConfig
	if cfg.Temperature != nil {
		opts["temperature"] = *cfg.Temperature
	}
	if cfg.TopP != nil {
		opts["top_p"] = *cfg.TopP
	}
	if cfg.TopK != nil {
		opts["top_k"] = *cfg.TopK
	}
	if cfg.MaxOutputTokens != 0 {
		opts["num_predict"] = cfg.MaxOutputTokens
	}
	if len(cfg.StopSequences) > 0 {
		opts["stop"] = cfg.StopSequences
	}
	if cfg.Seed != nil {
		opts["seed"] = *cfg.Seed
	}
	if cfg.PresencePenalty != nil {
		opts["presence_penalty"] = *cfg.PresencePenalty
	}
	if cfg.FrequencyPenalty != nil {
		opts["frequency_penalty"] = *cfg.FrequencyPenalty
	}

	// Note: Additional Ollama-specific options like num_ctx, repeat_penalty,
	// repeat_last_n, tfs_z, mirostat, etc. are not directly mapped from
	// genai.GenerateContentConfig. Users can pass these through custom
	// configuration if needed by extending OllamaConfig in the future.

	return opts
}

// generate performs a non-streaming generation.
func (o *ollamaModel) generate(ctx context.Context, req *api.ChatRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		streamOff := false
		req.Stream = &streamOff

		var finalResp *api.ChatResponse

		err := o.client.Chat(ctx, req, func(resp api.ChatResponse) error {
			finalResp = &resp
			return nil
		})

		if err != nil {
			yield(nil, formatOllamaError(err))
			return
		}

		if finalResp == nil {
			yield(nil, fmt.Errorf("no response from ollama"))
			return
		}

		llmResp := convertResponse(finalResp, true)
		yield(llmResp, nil)
	}
}

// generateStream performs a streaming generation.
func (o *ollamaModel) generateStream(ctx context.Context, req *api.ChatRequest) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		streamOn := true
		req.Stream = &streamOn

		var accumulatedToolCalls []api.ToolCall
		var lastResp *api.ChatResponse

		err := o.client.Chat(ctx, req, func(resp api.ChatResponse) error {
			lastResp = &resp

			// Accumulate tool calls
			if len(resp.Message.ToolCalls) > 0 {
				accumulatedToolCalls = append(accumulatedToolCalls, resp.Message.ToolCalls...)
			}

			// Yield partial response for text streaming
			if resp.Message.Content != "" {
				partialResp := &model.LLMResponse{
					Content: &genai.Content{
						Role: "model",
						Parts: []*genai.Part{
							{Text: resp.Message.Content},
						},
					},
					Partial:      !resp.Done,
					TurnComplete: resp.Done,
				}

				if !yield(partialResp, nil) {
					return fmt.Errorf("consumer stopped")
				}
			}

			return nil
		})

		if err != nil {
			yield(nil, formatOllamaError(err))
			return
		}

		// If we had tool calls, yield a final response with them
		if len(accumulatedToolCalls) > 0 && lastResp != nil {
			content, hasThinking := convertMessageToContent(&lastResp.Message)

			finalResp := &model.LLMResponse{
				Content:      content,
				TurnComplete: true,
				UsageMetadata: &genai.GenerateContentResponseUsageMetadata{
					PromptTokenCount:     int32(lastResp.PromptEvalCount),
					CandidatesTokenCount: int32(lastResp.EvalCount),
					TotalTokenCount:      int32(lastResp.PromptEvalCount + lastResp.EvalCount),
				},
			}

			// Set thinking token count if present
			if hasThinking {
				finalResp.UsageMetadata.ThoughtsTokenCount = 0 // Ollama doesn't provide separate count
			}

			// Add tool calls to content
			for _, tc := range accumulatedToolCalls {
				args := tc.Function.Arguments.ToMap()
				finalResp.Content.Parts = append(finalResp.Content.Parts, &genai.Part{
					FunctionCall: &genai.FunctionCall{
						Name: tc.Function.Name,
						Args: args,
					},
				})
			}

			// Add timing metrics
			if lastResp.TotalDuration > 0 || lastResp.LoadDuration > 0 {
				finalResp.CustomMetadata = map[string]any{
					"ollama_total_duration":       lastResp.TotalDuration,
					"ollama_load_duration":        lastResp.LoadDuration,
					"ollama_prompt_eval_duration": lastResp.PromptEvalDuration,
					"ollama_eval_duration":        lastResp.EvalDuration,
					"ollama_eval_count":           lastResp.EvalCount,
					"ollama_prompt_eval_count":    lastResp.PromptEvalCount,
				}
			}

			yield(finalResp, nil)
		}
	}
}

// convertResponse converts an Ollama ChatResponse to an ADK LLMResponse.
func convertResponse(resp *api.ChatResponse, turnComplete bool) *model.LLMResponse {
	content, hasThinking := convertMessageToContent(&resp.Message)

	llmResp := &model.LLMResponse{
		Content:      content,
		TurnComplete: turnComplete,
	}

	// Set usage metadata
	usageMeta := &genai.GenerateContentResponseUsageMetadata{
		PromptTokenCount:     int32(resp.PromptEvalCount),
		CandidatesTokenCount: int32(resp.EvalCount),
		TotalTokenCount:      int32(resp.PromptEvalCount + resp.EvalCount),
	}

	// If thinking content was present, note it in metadata
	// Note: Ollama doesn't provide separate token counts for thinking,
	// so we include it as part of candidates count
	if hasThinking {
		usageMeta.ThoughtsTokenCount = 0 // Would need separate count from Ollama
	}

	llmResp.UsageMetadata = usageMeta

	// Add timing metrics to custom metadata
	if resp.TotalDuration > 0 || resp.LoadDuration > 0 {
		llmResp.CustomMetadata = map[string]any{
			"ollama_total_duration":       resp.TotalDuration,        // nanoseconds
			"ollama_load_duration":        resp.LoadDuration,         // nanoseconds
			"ollama_prompt_eval_duration": resp.PromptEvalDuration,   // nanoseconds
			"ollama_eval_duration":        resp.EvalDuration,         // nanoseconds
			"ollama_eval_count":           resp.EvalCount,            // tokens
			"ollama_prompt_eval_count":    resp.PromptEvalCount,      // tokens
		}
	}

	// Map done reason to finish reason
	if resp.Done {
		switch resp.DoneReason {
		case "stop":
			llmResp.FinishReason = genai.FinishReasonStop
		case "length":
			llmResp.FinishReason = genai.FinishReasonMaxTokens
		default:
			if resp.DoneReason != "" {
				llmResp.FinishReason = genai.FinishReasonOther
			}
		}
	}

	return llmResp
}

// convertMessageToContent converts an Ollama Message to genai.Content.
// Returns the content and a boolean indicating if thinking content was present.
func convertMessageToContent(msg *api.Message) (*genai.Content, bool) {
	content := &genai.Content{
		Role:  "model",
		Parts: []*genai.Part{},
	}

	hasThinking := false

	// Add text content
	if msg.Content != "" {
		content.Parts = append(content.Parts, &genai.Part{
			Text: msg.Content,
		})
	}

	// Add thinking content if present - differentiate it with metadata
	// Note: genai.Part doesn't have a direct "thinking" field, so we include
	// it as text but could be separated in a future enhancement
	if msg.Thinking != "" {
		hasThinking = true
		// Prepend thinking content with a marker so consumers can identify it
		thinkingText := "[Thinking] " + msg.Thinking
		content.Parts = append([]*genai.Part{{Text: thinkingText}}, content.Parts...)
	}

	// Convert tool calls to function calls
	for _, tc := range msg.ToolCalls {
		args := tc.Function.Arguments.ToMap()
		content.Parts = append(content.Parts, &genai.Part{
			FunctionCall: &genai.FunctionCall{
				Name: tc.Function.Name,
				Args: args,
			},
		})
	}

	return content, hasThinking
}

// parseKeepAlive parses a keep_alive string into a time.Duration.
// Accepts duration strings like "5m", "30s" or numeric strings representing seconds.
func parseKeepAlive(keepAlive string) (time.Duration, error) {
	// Try parsing as duration string first (e.g., "5m", "30s")
	if d, err := time.ParseDuration(keepAlive); err == nil {
		return d, nil
	}

	// Try parsing as number of seconds
	if seconds, err := strconv.ParseFloat(keepAlive, 64); err == nil {
		return time.Duration(seconds * float64(time.Second)), nil
	}

	return 0, fmt.Errorf("invalid keep_alive format: %s", keepAlive)
}

// convertSchemaToFormat converts a genai.Schema to JSON format for Ollama.
func convertSchemaToFormat(schema *genai.Schema) (json.RawMessage, error) {
	if schema == nil {
		return nil, fmt.Errorf("schema is nil")
	}

	// Convert genai.Schema to a map representation
	schemaMap := schemaToMap(schema)

	// Marshal to JSON
	jsonBytes, err := json.Marshal(schemaMap)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	return json.RawMessage(jsonBytes), nil
}

// schemaToMap converts a genai.Schema to a map[string]any for JSON serialization.
func schemaToMap(schema *genai.Schema) map[string]any {
	if schema == nil {
		return nil
	}

	result := map[string]any{
		"type": string(schema.Type),
	}

	if schema.Description != "" {
		result["description"] = schema.Description
	}

	if len(schema.Enum) > 0 {
		result["enum"] = schema.Enum
	}

	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}

	if len(schema.Properties) > 0 {
		props := make(map[string]any)
		for name, prop := range schema.Properties {
			props[name] = schemaToMap(prop)
		}
		result["properties"] = props
	}

	if schema.Items != nil {
		result["items"] = schemaToMap(schema.Items)
	}

	return result
}

// formatOllamaError formats an Ollama API error with additional context.
// It extracts HTTP status codes and error messages from api.StatusError.
func formatOllamaError(err error) error {
	if err == nil {
		return nil
	}

	// Check if it's a StatusError from the Ollama API
	var statusErr api.StatusError
	if ok := errors.As(err, &statusErr); ok {
		// Format error with HTTP status code context
		switch statusErr.StatusCode {
		case 400:
			return fmt.Errorf("bad request (400): %s", statusErr.ErrorMessage)
		case 404:
			return fmt.Errorf("model not found (404): %s", statusErr.ErrorMessage)
		case 429:
			return fmt.Errorf("rate limit exceeded (429): %s", statusErr.ErrorMessage)
		case 500:
			return fmt.Errorf("internal server error (500): %s", statusErr.ErrorMessage)
		case 502:
			return fmt.Errorf("bad gateway (502): %s", statusErr.ErrorMessage)
		default:
			return fmt.Errorf("ollama API error (%d): %s", statusErr.StatusCode, statusErr.ErrorMessage)
		}
	}

	// Return wrapped error for other error types
	return fmt.Errorf("ollama error: %w", err)
}
