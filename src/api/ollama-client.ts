import {
	ModelApi,
	BaseModelRequest,
	ExtendedModelRequest,
	ModelResponse,
	ToolCall,
	StreamCallback,
	StreamingModelResponse,
} from './interfaces/model-api';
import { GeminiPrompts } from '../prompts';
import type ObsidianGemini from '../main';
import { getDefaultModelForRole } from '../models';

export interface OllamaClientConfig {
	baseUrl: string;
	model?: string;
	temperature?: number;
	topP?: number;
	maxOutputTokens?: number;
	streamingEnabled?: boolean;
}

export class OllamaClient implements ModelApi {
	private config: OllamaClientConfig;
	private prompts: GeminiPrompts;
	private plugin?: ObsidianGemini;

	constructor(config: OllamaClientConfig, prompts?: GeminiPrompts, plugin?: ObsidianGemini) {
		this.config = {
			temperature: 0.7,
			topP: 0.95,
			streamingEnabled: true,
			...config,
		};
		this.plugin = plugin;
		this.prompts = prompts || new GeminiPrompts(plugin);
	}

	async generateModelResponse(request: BaseModelRequest | ExtendedModelRequest): Promise<ModelResponse> {
		const payload = await this.buildRequestPayload(request, false);
		const url = `${this.config.baseUrl}/api/chat`;

		try {
			const response = await fetch(url, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(payload),
			});

			if (!response.ok) {
				throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
			}

			const data = await response.json();
			return this.extractModelResponse(data);
		} catch (error) {
			this.plugin?.logger.error('[OllamaClient] Error generating content:', error);
			throw error;
		}
	}

	generateStreamingResponse(
		request: BaseModelRequest | ExtendedModelRequest,
		onChunk: StreamCallback
	): StreamingModelResponse {
		let cancelled = false;
		let accumulatedText = '';
		let toolCalls: ToolCall[] | undefined;
		let abortController = new AbortController();

		const complete = (async (): Promise<ModelResponse> => {
			const payload = await this.buildRequestPayload(request, true);
			const url = `${this.config.baseUrl}/api/chat`;

			try {
				const response = await fetch(url, {
					method: 'POST',
					headers: { 'Content-Type': 'application/json' },
					body: JSON.stringify(payload),
					signal: abortController.signal,
				});

				if (!response.ok) {
					throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
				}

				if (!response.body) {
					throw new Error('ReadableStream not supported in this environment');
				}

				const reader = response.body.getReader();
				const decoder = new TextDecoder();
				let buffer = '';

				while (true) {
					if (cancelled) break;
					const { done, value } = await reader.read();
					if (done) break;

					buffer += decoder.decode(value, { stream: true });
					const lines = buffer.split('\n');
					buffer = lines.pop() || '';

					for (const line of lines) {
						if (!line.trim()) continue;
						try {
							const data = JSON.parse(line);

							if (data.message) {
								if (data.message.content) {
									accumulatedText += data.message.content;
									onChunk({ text: data.message.content });
								}

								if (data.message.tool_calls) {
									if (!toolCalls) toolCalls = [];
									for (const call of data.message.tool_calls) {
										toolCalls.push({
											name: call.function.name,
											arguments: call.function.arguments,
										});
									}
								}
							}
						} catch (e) {
							this.plugin?.logger.warn('Error parsing JSON from stream line:', line);
						}
					}
				}
			} catch (error: any) {
				if (error.name === 'AbortError' || cancelled) {
					// Expected abort
				} else {
					this.plugin?.logger.error('[OllamaClient] Streaming error:', error);
					throw error;
				}
			}

			return {
				markdown: accumulatedText,
				rendered: '',
				...(toolCalls && { toolCalls }),
			};
		})();

		return {
			complete,
			cancel: () => {
				cancelled = true;
				abortController.abort();
			},
		};
	}

	private async buildRequestPayload(request: BaseModelRequest | ExtendedModelRequest, stream: boolean) {
		const isExtended = 'userMessage' in request;
		const model = request.model || this.config.model || getDefaultModelForRole('chat');

		const messages: any[] = [];
		let systemInstruction = '';

		if (isExtended) {
			const extReq = request as ExtendedModelRequest;

			let agentsMemory: string | null = null;
			if (this.plugin?.agentsMemory) {
				try {
					agentsMemory = await this.plugin.agentsMemory.read();
				} catch (error) {
					this.plugin?.logger.warn('Failed to load AGENTS.md:', error);
				}
			}

			let availableSkills: { name: string; description: string }[] = [];
			if (this.plugin?.skillManager) {
				try {
					availableSkills = await this.plugin.skillManager.getSkillSummaries();
				} catch (error) {
					this.plugin?.logger.warn('Failed to load skill summaries:', error);
				}
			}

			systemInstruction = this.prompts.getSystemPromptWithCustom(
				extReq.availableTools,
				extReq.customPrompt,
				agentsMemory,
				availableSkills
			);

			if (extReq.prompt && !extReq.customPrompt?.overrideSystemPrompt) {
				systemInstruction += '\n\n' + extReq.prompt;
			}

			messages.push({ role: 'system', content: systemInstruction });

			if (extReq.conversationHistory?.length) {
				// Track the most recent tool call id by function name so we can correlate responses.
				const lastToolCallId: Record<string, string> = {};
				let toolCallCounter = 0;

				for (const entry of extReq.conversationHistory) {
					if ('role' in entry && 'parts' in entry) {
						// Gemini format — map tool-call and tool-result parts to Ollama format
						const role = entry.role === 'model' ? 'assistant' : entry.role;
						const parts: any[] = entry.parts;

						// Check for functionCall parts (assistant calling a tool)
						const functionCallParts = parts.filter((p: any) => p.functionCall);
						// Check for functionResponse parts (tool results, stored in user turns)
						const functionResponseParts = parts.filter((p: any) => p.functionResponse);
						const textContent = parts
							.filter((p: any) => p.text)
							.map((p: any) => p.text)
							.join('');

						if (functionCallParts.length > 0) {
							// Assistant message with tool calls — include id for each call
							const toolCalls = functionCallParts.map((p: any) => {
								const id = `call_${p.functionCall.name}_${toolCallCounter++}`;
								lastToolCallId[p.functionCall.name] = id;
								return {
									id,
									type: 'function' as const,
									function: {
										name: p.functionCall.name,
										arguments: JSON.stringify(p.functionCall.args || {}),
									},
								};
							});
							messages.push({
								role: 'assistant',
								content: textContent,
								tool_calls: toolCalls,
							});
						} else if (functionResponseParts.length > 0) {
							// Each functionResponse maps to a separate tool message
							for (const p of functionResponseParts) {
								const toolCallId = lastToolCallId[p.functionResponse.name] ?? `call_${p.functionResponse.name}_0`;
								messages.push({
									role: 'tool',
									tool_call_id: toolCallId,
									content:
										typeof p.functionResponse.response === 'string'
											? p.functionResponse.response
											: JSON.stringify(p.functionResponse.response),
								});
							}
							// If there's additional text in the same turn, emit it as a user message
							if (textContent) {
								messages.push({ role: 'user', content: textContent });
							}
						} else {
							messages.push({ role, content: textContent });
						}
					} else if ('role' in entry && 'text' in entry) {
						messages.push({ role: entry.role === 'model' ? 'assistant' : 'user', content: entry.text });
					} else if ('role' in entry && 'message' in entry) {
						messages.push({ role: entry.role === 'model' ? 'assistant' : 'user', content: entry.message });
					}
				}
			}

			let userContent = extReq.userMessage || '';
			const images: string[] = [];

			const allAttachments = [...(extReq.inlineAttachments || []), ...(extReq.imageAttachments || [])];
			for (const attachment of allAttachments) {
				images.push(attachment.base64);
			}

			if (userContent || images.length > 0) {
				const msg: any = { role: 'user', content: userContent };
				if (images.length > 0) {
					msg.images = images;
				}
				messages.push(msg);
			}
		} else {
			messages.push({ role: 'user', content: request.prompt || '' });
		}

		const payload: any = {
			model,
			messages,
			stream,
			options: {
				temperature: request.temperature ?? this.config.temperature,
				top_p: request.topP ?? this.config.topP,
			},
		};

		if (isExtended && (request as ExtendedModelRequest).availableTools?.length) {
			const tools = (request as ExtendedModelRequest).availableTools!;
			payload.tools = tools.map((tool) => ({
				type: 'function',
				function: {
					name: tool.name,
					description: tool.description,
					parameters: {
						type: 'object',
						properties: tool.parameters.properties || {},
						required: tool.parameters.required || [],
					},
				},
			}));
		}

		return payload;
	}

	private extractModelResponse(data: any): ModelResponse {
		let markdown = data.message?.content || '';
		let toolCalls: ToolCall[] | undefined;

		if (data.message?.tool_calls) {
			toolCalls = data.message.tool_calls.map((call: any) => ({
				name: call.function.name,
				arguments: call.function.arguments,
			}));
		}

		return {
			markdown,
			rendered: '',
			...(toolCalls && { toolCalls }),
		};
	}

	async generateImage(_prompt?: string, _model?: string): Promise<string> {
		throw new Error('Image generation is not supported by Ollama yet.');
	}
}
