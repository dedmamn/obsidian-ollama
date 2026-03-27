/**
 * API module for Gemini AI integration
 */

// Re-export the interfaces
export type {
	ModelApi,
	ModelResponse,
	BaseModelRequest,
	ExtendedModelRequest,
	InlineDataPart,
	ImagePart,
	ToolCall,
	ToolDefinition,
} from './interfaces/model-api';

// Export the simplified factory
export { OllamaClientFactory, ModelUseCase } from './simple-factory';

// Export the client
export { OllamaClient } from './ollama-client';
export type { OllamaClientConfig } from './ollama-client';
