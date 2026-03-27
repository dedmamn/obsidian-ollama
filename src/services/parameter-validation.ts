import { OllamaModel } from './model-discovery';

export interface ParameterRanges {
	temperature: {
		min: number;
		max: number;
		step: number;
	};
	topP: {
		min: number;
		max: number;
		step: number;
	};
}

export interface ModelParameterInfo {
	modelName: string;
	maxTemperature?: number;
	topP?: number;
	topK?: number;
}

export class ParameterValidationService {
	/**
	 * Default fallback ranges when no model information is available
	 */
	private static readonly DEFAULT_RANGES: ParameterRanges = {
		temperature: { min: 0, max: 2, step: 0.1 },
		topP: { min: 0, max: 1, step: 0.01 },
	};

	/**
	 * Get parameter ranges based on discovered model information
	 */
	static getParameterRanges(discoveredModels: OllamaModel[]): ParameterRanges {
		if (!discoveredModels || discoveredModels.length === 0) {
			return this.DEFAULT_RANGES;
		}

		// Find the maximum temperature across all models
		const maxTemperatures = [] as number[];

		const maxTemp =
			maxTemperatures.length > 0
				? maxTemperatures.reduce((max, temp) => Math.max(max, temp), 0)
				: this.DEFAULT_RANGES.temperature.max;

		return {
			temperature: {
				min: 0,
				max: Math.max(maxTemp, 1), // Ensure at least 1 as minimum useful range
				step: 0.1,
			},
			topP: {
				min: 0,
				max: 1, // topP is always 0-1 for Gemini models
				step: 0.01,
			},
		};
	}

	/**
	 * Get parameter information for specific models
	 */
	static getModelParameterInfo(discoveredModels: OllamaModel[]): ModelParameterInfo[] {
		return discoveredModels.map((model) => ({
			modelName: model.name,
		}));
	}

	/**
	 * Validate temperature value against model capabilities
	 */
	static validateTemperature(
		value: number,
		_modelName?: string,
		discoveredModels: OllamaModel[] = []
	): {
		isValid: boolean;
		adjustedValue?: number;
		warning?: string;
	} {
		// Then check against global ranges
		const ranges = this.getParameterRanges(discoveredModels);

		if (value < ranges.temperature.min || value > ranges.temperature.max) {
			const adjustedValue = Math.max(ranges.temperature.min, Math.min(ranges.temperature.max, value));
			return {
				isValid: false,
				adjustedValue,
				warning: `Temperature ${value} is outside valid range [${ranges.temperature.min}, ${ranges.temperature.max}]. Adjusted to ${adjustedValue}.`,
			};
		}

		return { isValid: true };
	}

	/**
	 * Validate topP value against model capabilities
	 */
	static validateTopP(
		value: number,
		_modelName?: string,
		discoveredModels: OllamaModel[] = []
	): {
		isValid: boolean;
		adjustedValue?: number;
		warning?: string;
	} {
		const ranges = this.getParameterRanges(discoveredModels);

		if (value < ranges.topP.min || value > ranges.topP.max) {
			const adjustedValue = Math.max(ranges.topP.min, Math.min(ranges.topP.max, value));
			return {
				isValid: false,
				adjustedValue,
				warning: `Top P ${value} is outside valid range [${ranges.topP.min}, ${ranges.topP.max}]. Adjusted to ${adjustedValue}.`,
			};
		}

		return { isValid: true };
	}

	/**
	 * Get user-friendly parameter information for display in settings
	 */
	static getParameterDisplayInfo(discoveredModels: OllamaModel[]): {
		temperature: string;
		topP: string;
		hasModelData: boolean;
	} {
		const ranges = this.getParameterRanges(discoveredModels);
		const hasModelData = discoveredModels && discoveredModels.length > 0;

		const uniqueTopPValues: number[] = [];

		const topPInfo =
			uniqueTopPValues.length > 0
				? `Range: ${ranges.topP.min} to ${ranges.topP.max} (model defaults: ${uniqueTopPValues.join(', ')})`
				: `Range: ${ranges.topP.min} to ${ranges.topP.max}`;

		return {
			temperature: `Range: ${ranges.temperature.min} to ${ranges.temperature.max}`,
			topP: topPInfo,
			hasModelData,
		};
	}
}
