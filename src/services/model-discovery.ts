import ObsidianGemini from '../main';

export interface OllamaModel {
	name: string;
	model: string;
	modified_at: string;
	size: number;
	digest: string;
	details: {
		parent_model: string;
		format: string;
		family: string;
		families: string[];
		parameter_size: string;
		quantization_level: string;
	};
}

export interface ModelDiscoveryResult {
	models: OllamaModel[];
	lastUpdated: number;
	success: boolean;
	error?: string;
}

export class ModelDiscoveryService {
	private plugin: ObsidianGemini;
	private cache: ModelDiscoveryResult | null = null;
	private readonly CACHE_DURATION = 1 * 60 * 60 * 1000; // 1 hour

	constructor(plugin: ObsidianGemini) {
		this.plugin = plugin;
	}

	async discoverModels(forceRefresh = false): Promise<ModelDiscoveryResult> {
		if (!forceRefresh && this.cache && this.isCacheValid()) {
			return this.cache;
		}

		try {
			const models = await this.fetchModelsFromAPI();
			const result: ModelDiscoveryResult = {
				models,
				lastUpdated: Date.now(),
				success: true,
			};

			this.cache = result;
			await this.persistCache(result);
			return result;
		} catch (error) {
			const result: ModelDiscoveryResult = {
				models: [],
				lastUpdated: Date.now(),
				success: false,
				error: error instanceof Error ? error.message : 'Unknown error',
			};
			return this.cache || result;
		}
	}

	private async fetchModelsFromAPI(): Promise<OllamaModel[]> {
		const baseUrl = this.plugin.settings.ollamaBaseUrl;
		if (!baseUrl) {
			throw new Error('Ollama Base URL not configured');
		}

		const url = new URL(`${baseUrl}/api/tags`);
		const response = await fetch(url.toString());
		if (!response.ok) {
			throw new Error(`API request failed: ${response.status} ${response.statusText}`);
		}

		const data = await response.json();
		return data.models || [];
	}

	private isCacheValid(): boolean {
		return !!(this.cache && Date.now() - this.cache.lastUpdated < this.CACHE_DURATION);
	}

	private async persistCache(result: ModelDiscoveryResult): Promise<void> {
		const data = (await this.plugin.loadData()) || {};
		data.modelDiscoveryCache = result;
		await this.plugin.saveData(data);
	}

	async loadCache(): Promise<void> {
		const data = (await this.plugin.loadData()) || {};
		this.cache = data.modelDiscoveryCache || null;
	}

	clearCache(): void {
		this.cache = null;
	}

	getCacheInfo(): { hasCache: boolean; isValid: boolean; lastUpdated?: number } {
		return {
			hasCache: !!this.cache,
			isValid: this.isCacheValid(),
			lastUpdated: this.cache?.lastUpdated,
		};
	}
}
