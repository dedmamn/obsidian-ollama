import { Tool, ToolResult, ToolExecutionContext } from './types';
import { ToolCategory } from '../types/agent';
import { ToolClassification } from '../types/tool-policy';

/**
 * Google Search tool that uses a separate model instance with search grounding
 */
export class GoogleSearchTool implements Tool {
	name = 'google_search';
	displayName = 'Google Search';
	category = ToolCategory.READ_ONLY;
	classification = ToolClassification.EXTERNAL;
	description =
		"Search Google for current, up-to-date information from the web using Google's Search Grounding feature. Returns AI-generated answer with inline citations and source links. Use this to find recent news, facts, statistics, or any information that might have changed since the AI model's training cutoff. Results include structured citations with URLs, titles, and snippets from authoritative web sources.";

	parameters = {
		type: 'object' as const,
		properties: {
			query: {
				type: 'string' as const,
				description: 'The search query to send to Google',
			},
		},
		required: ['query'],
	};

	getProgressDescription(params: { query: string }): string {
		if (params.query) {
			// Truncate long queries
			const query = params.query.length > 30 ? params.query.substring(0, 27) + '...' : params.query;
			return `Searching Google for "${query}"`;
		}
		return 'Searching Google';
	}

	async execute(_params: { query: string }, _context: ToolExecutionContext): Promise<ToolResult> {
		return {
			success: false,
			error: 'Google Search is not supported with Ollama. This feature requires the Google Gemini API.',
		};
	}
}

/**
 * Get Google Search tool
 */
export function getGoogleSearchTool(): Tool {
	return new GoogleSearchTool();
}
