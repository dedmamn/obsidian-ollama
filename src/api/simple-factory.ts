/**
 * Simplified factory for creating Gemini API clients
 *
 * Replaces the complex ApiFactory and ModelFactory with a single,
 * straightforward approach focused solely on Gemini.
 */

import { OllamaClient, OllamaClientConfig } from './ollama-client';
import { ModelApi } from './interfaces/model-api';
import { GeminiPrompts } from '../prompts';
import { RetryDecorator } from './retry-decorator';
import { getDefaultModelForRole } from '../models';
import type ObsidianGemini from '../main';

/**
 * Model use cases for the plugin
 */
export enum ModelUseCase {
	CHAT = 'chat',
	SUMMARY = 'summary',
	COMPLETIONS = 'completions',
	REWRITE = 'rewrite',
	SEARCH = 'search',
}

/**
 * Simple factory for creating Gemini API clients
 */
export class OllamaClientFactory {
	/**
	 * Create a OllamaClient from plugin settings
	 *
	 * @param plugin - Plugin instance with settings
	 * @param useCase - The use case for this model (determines which model to use)
	 * @param overrides - Optional config overrides (for per-session settings)
	 * @returns Configured OllamaClient instance
	 */
	static createFromPlugin(
		plugin: ObsidianGemini,
		useCase: ModelUseCase,
		overrides?: Partial<OllamaClientConfig>
	): ModelApi {
		const settings = plugin.settings;

		// Determine which model to use based on use case
		let modelName: string;
		switch (useCase) {
			case ModelUseCase.CHAT:
				modelName = settings.chatModelName || getDefaultModelForRole('chat');
				break;
			case ModelUseCase.SUMMARY:
				modelName = settings.summaryModelName || getDefaultModelForRole('summary');
				break;
			case ModelUseCase.COMPLETIONS:
				modelName = settings.completionsModelName || getDefaultModelForRole('completions');
				break;
			case ModelUseCase.REWRITE:
				// Rewrite uses chat model
				modelName = settings.chatModelName || getDefaultModelForRole('chat');
				break;
			case ModelUseCase.SEARCH:
				// Search uses chat model
				modelName = settings.chatModelName || getDefaultModelForRole('chat');
				break;
			default:
				modelName = getDefaultModelForRole('chat');
		}

		// Build config
		const config: OllamaClientConfig = {
			baseUrl: plugin.ollamaUrl,
			model: modelName,
			temperature: settings.temperature ?? 1.0,
			topP: settings.topP ?? 0.95,
			streamingEnabled: settings.streamingEnabled ?? true,
			...overrides,
		};

		// Create prompts instance with plugin reference so it can access settings
		const prompts = new GeminiPrompts(plugin);

		// Create client
		const client = new OllamaClient(config, prompts, plugin);

		// Wrap with retry decorator
		const retryConfig = {
			maxRetries: settings.maxRetries ?? 3,
			initialBackoffDelay: settings.initialBackoffDelay ?? 1000,
		};

		return new RetryDecorator(client, retryConfig, plugin.logger);
	}

	/**
	 * Create a OllamaClient with custom configuration
	 *
	 * @param config - Complete client configuration
	 * @param prompts - Optional prompts instance
	 * @param plugin - Optional plugin instance
	 * @returns Configured OllamaClient instance wrapped with retry logic
	 */
	static createCustom(config: OllamaClientConfig, prompts?: GeminiPrompts, plugin?: ObsidianGemini): ModelApi {
		const client = new OllamaClient(config, prompts, plugin);

		// Use retry config from plugin settings if available, otherwise use defaults
		const retryConfig = plugin
			? {
					maxRetries: plugin.settings.maxRetries ?? 3,
					initialBackoffDelay: plugin.settings.initialBackoffDelay ?? 1000,
				}
			: {
					maxRetries: 3,
					initialBackoffDelay: 1000,
				};

		return new RetryDecorator(client, retryConfig, plugin?.logger);
	}

	/**
	 * Create a chat model with optional session-specific overrides
	 *
	 * @param plugin - Plugin instance
	 * @param sessionConfig - Optional session-level config (model, temperature, topP)
	 * @returns Configured OllamaClient for chat
	 */
	static createChatModel(
		plugin: ObsidianGemini,
		sessionConfig?: { model?: string; temperature?: number; topP?: number }
	): ModelApi {
		const overrides: Partial<OllamaClientConfig> = {};

		if (sessionConfig) {
			// Session config takes precedence
			if (sessionConfig.temperature !== undefined) {
				overrides.temperature = sessionConfig.temperature;
			}
			if (sessionConfig.topP !== undefined) {
				overrides.topP = sessionConfig.topP;
			}
			// Note: model override is handled at request time via session.modelConfig
		}

		return this.createFromPlugin(plugin, ModelUseCase.CHAT, overrides);
	}

	/**
	 * Create a summary model
	 *
	 * @param plugin - Plugin instance
	 * @returns Configured OllamaClient for summaries
	 */
	static createSummaryModel(plugin: ObsidianGemini): ModelApi {
		return this.createFromPlugin(plugin, ModelUseCase.SUMMARY);
	}

	/**
	 * Create a completions model
	 *
	 * @param plugin - Plugin instance
	 * @returns Configured OllamaClient for completions
	 */
	static createCompletionsModel(plugin: ObsidianGemini): ModelApi {
		return this.createFromPlugin(plugin, ModelUseCase.COMPLETIONS);
	}

	/**
	 * Create a rewrite model
	 *
	 * @param plugin - Plugin instance
	 * @returns Configured OllamaClient for rewriting
	 */
	static createRewriteModel(plugin: ObsidianGemini): ModelApi {
		return this.createFromPlugin(plugin, ModelUseCase.REWRITE);
	}

	/**
	 * Create a search model
	 *
	 * @param plugin - Plugin instance
	 * @returns Configured OllamaClient for search operations
	 */
	static createSearchModel(plugin: ObsidianGemini): ModelApi {
		return this.createFromPlugin(plugin, ModelUseCase.SEARCH);
	}
}
