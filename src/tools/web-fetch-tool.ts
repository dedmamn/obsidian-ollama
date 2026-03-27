import { Tool, ToolResult, ToolExecutionContext } from './types';
import { ToolCategory } from '../types/agent';
import { ToolClassification } from '../types/tool-policy';
import type ObsidianGemini from '../main';

import { requestUrlWithRetry } from '../utils/proxy-fetch';
import TurndownService from 'turndown';
import { decodeHtmlEntities } from '../utils/html-entities';

/**
 * Web fetch tool using Google's URL Context feature
 * This allows the model to fetch and analyze content from URLs
 *
 * Note: URL context is automatically recognized when a URL is present in the prompt.
 * The model will fetch and analyze the content at the URL.
 */
export class WebFetchTool implements Tool {
	name = 'fetch_url';
	displayName = 'Fetch URL';
	category = ToolCategory.READ_ONLY;
	classification = ToolClassification.EXTERNAL;
	description =
		"Fetch and analyze content from a specific URL using Google's URL Context feature and AI. Provide a URL and a query describing what information to extract or questions to answer about the page content. The AI will read the page and provide a targeted analysis based on your query. Returns the analyzed content, URL metadata, and fetch timestamp. Falls back to direct HTTP fetch if URL Context fails. Use this to extract specific information from web pages, documentation, articles, or any publicly accessible URL.";

	parameters = {
		type: 'object' as const,
		properties: {
			url: {
				type: 'string' as const,
				description: 'The URL to fetch and analyze',
			},
			query: {
				type: 'string' as const,
				description: 'What information to extract or questions to answer about the content',
			},
		},
		required: ['url', 'query'],
	};

	getProgressDescription(params: { url: string }): string {
		if (params.url) {
			// Extract domain for brevity
			try {
				const domain = new URL(params.url).hostname.replace('www.', '');
				return `Fetching from ${domain}`;
			} catch {
				return 'Fetching web page';
			}
		}
		return 'Fetching web page';
	}

	async execute(params: { url: string; query: string }, context: ToolExecutionContext): Promise<ToolResult> {
		const plugin = context.plugin as InstanceType<typeof ObsidianGemini>;

		try {
			// Validate URL
			const urlObj = new URL(params.url);
			if (!['http:', 'https:'].includes(urlObj.protocol)) {
				return {
					success: false,
					error: 'Only HTTP and HTTPS URLs are supported',
				};
			}

			// URL Context (AI-powered extraction) is not available with Ollama.
			// Fall back to direct HTTP fetch + markdown conversion.
			return await this.fallbackFetch(params, plugin);
		} catch (error) {
			plugin.logger.error('Web fetch error:', error);

			if (error instanceof TypeError && error.message.includes('Failed to construct')) {
				return {
					success: false,
					error: `Invalid URL format: ${params.url}`,
				};
			}

			return {
				success: false,
				error: `Failed to fetch URL: ${error instanceof Error ? error.message : 'Unknown error'}`,
			};
		}
	}

	/**
	 * Direct HTTP fetch with HTML-to-Markdown conversion.
	 */
	private async fallbackFetch(
		params: { url: string; query: string },
		plugin: InstanceType<typeof ObsidianGemini>
	): Promise<ToolResult> {
		try {
			const response = await requestUrlWithRetry({
				url: params.url,
				method: 'GET',
				headers: {
					'User-Agent': 'Mozilla/5.0 (compatible; ObsidianGemini/1.0)',
				},
			});

			if (response.status !== 200) {
				return {
					success: false,
					error: `HTTP ${response.status}: ${response.text || 'Failed to fetch URL'}`,
				};
			}

			const rawHtml = response.text;

			// Extract title before conversion
			const titleMatch = rawHtml.match(/<title[^>]*>([^<]+)<\/title>/i);
			const title = titleMatch ? decodeHtmlEntities(titleMatch[1].trim()) : params.url;

			// Configure turndown to strip scripts and styles, convert to clean Markdown
			const turndownService = new TurndownService({
				headingStyle: 'atx',
				codeBlockStyle: 'fenced',
			});

			// Remove script, style, nav, and footer elements entirely
			turndownService.remove(['script', 'style', 'nav', 'footer', 'noscript']);

			let content = turndownService.turndown(rawHtml);

			// Truncate if too long
			if (content.length > 10000) {
				content = content.substring(0, 10000) + '\n\n[Content truncated...]';
			}

			return {
				success: true,
				data: {
					url: params.url,
					query: params.query,
					content,
					title,
					fallbackMethod: true,
					fetchedAt: new Date().toISOString(),
				},
			};
		} catch (error) {
			plugin.logger.error('Fallback fetch error:', error);
			return {
				success: false,
				error: `Fallback fetch failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
			};
		}
	}
}
