import { GeminiModel, ModelRole } from '../models';
import { OllamaModel } from './model-discovery';

export class ModelMapper {
	static mapToGeminiModels(ollamaModels: OllamaModel[]): GeminiModel[] {
		const mappedModels = ollamaModels.map((model) => {
			const value = model.name;
			const supportsImageGeneration = value.toLowerCase().includes('llava');

			return {
				value,
				label: model.name,
				defaultForRoles: this.inferDefaultRoles(model),
				supportsImageGeneration,
			};
		});

		return this.deduplicateModels(mappedModels);
	}

	static deduplicateModels(models: GeminiModel[]): GeminiModel[] {
		const seen = new Map<string, GeminiModel>();
		for (const model of models) {
			seen.set(model.value, model);
		}
		return Array.from(seen.values());
	}

	private static inferDefaultRoles(model: OllamaModel): ModelRole[] {
		const modelId = model.name.toLowerCase();
		const roles: ModelRole[] = [];

		if (modelId.includes('llava')) {
			roles.push('image');
		} else {
			roles.push('chat', 'summary', 'completions', 'rewrite');
		}

		return roles;
	}

	static mergeWithExistingModels(discoveredModels: GeminiModel[], existingModels: GeminiModel[]): GeminiModel[] {
		const existingMap = new Map(existingModels.map((model) => [model.value, model]));
		const currentDefaults = this.getCurrentDefaultModels(existingModels);

		const mergedModels = discoveredModels.map((discovered) => {
			const existing = existingMap.get(discovered.value);
			if (existing) {
				return {
					...discovered,
					defaultForRoles: existing.defaultForRoles,
					label: existing.label,
				};
			}
			return discovered;
		});

		return this.ensureRoleDefaults(mergedModels, currentDefaults);
	}

	private static getCurrentDefaultModels(existingModels: GeminiModel[]): { [role in ModelRole]?: string } {
		const defaults: { [role in ModelRole]?: string } = {};
		for (const role of ['chat', 'summary', 'completions', 'rewrite', 'image'] as ModelRole[]) {
			const defaultModel = existingModels.find((m) => m.defaultForRoles?.includes(role));
			if (defaultModel) {
				defaults[role] = defaultModel.value;
			}
		}
		return defaults;
	}

	private static ensureRoleDefaults(
		models: GeminiModel[],
		currentDefaults: { [role in ModelRole]?: string }
	): GeminiModel[] {
		const modelsMap = new Map(models.map((m) => [m.value, m]));

		for (const role of ['chat', 'summary', 'completions', 'rewrite', 'image'] as ModelRole[]) {
			const currentDefault = currentDefaults[role];
			const hasDefault = models.some((m) => m.defaultForRoles?.includes(role));

			if (!hasDefault) {
				if (currentDefault && modelsMap.has(currentDefault)) {
					const model = modelsMap.get(currentDefault)!;
					model.defaultForRoles = [...(model.defaultForRoles || []), role];
				} else {
					const bestMatch = models[0];
					if (bestMatch && !bestMatch.defaultForRoles?.includes(role)) {
						bestMatch.defaultForRoles = [...(bestMatch.defaultForRoles || []), role];
					}
				}
			}
		}

		return models;
	}

	static sortModelsByPreference(models: GeminiModel[]): GeminiModel[] {
		return [...models].sort((a, b) => a.label.localeCompare(b.label));
	}
}
