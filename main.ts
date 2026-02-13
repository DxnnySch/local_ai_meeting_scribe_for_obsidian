import { App, Modal, Notice, Plugin, PluginSettingTab, Setting, TFolder, normalizePath, setIcon } from 'obsidian';

interface Segment {
	start: number;
	end: number;
	text: string;
	speaker?: string;
}

interface StopResponse {
	segments: Segment[];
	speakers: string[];
	samples: Record<string, string>;
	audio_samples?: Record<string, string>;
}

interface SummarizeResponse {
	summary: string;
}

interface MicDevice {
	id: string;
	name: string;
}

interface LanguageOption {
	code: string;
	label: string;
}

interface LlmModelOption {
	id: string;
	name: string;
}

interface MicTestResponse {
	ok: boolean;
	device_name: string;
	rms: number;
	peak: number;
	error?: string | null;
}

type CaptureProfile = 'fast' | 'balanced' | 'pristine';
type AiQualityMode = 'efficient' | 'full_vram_12gb';

interface MeetingScribeSettings {
	serverBaseUrl: string;
	ollamaBaseUrl: string;
	defaultSummaryModel: string;
	defaultFolder: string;
	defaultMeetingLanguage: string;
	defaultTargetApps: string;
	defaultMicDeviceContains: string;
	defaultIncludeMicInput: boolean;
	defaultOwnSpeakerName: string;
	defaultAiQualityMode: AiQualityMode;
	autoSummarize: boolean;
	includeFullTranscript: boolean;
	expectedSpeakers: number;
	captureProfile: CaptureProfile;
	enableDiarization: boolean;
	preferLoopbackCapture: boolean;
}

interface SessionCaptureOptions {
	captureProfile: CaptureProfile;
	enableDiarization: boolean;
	preferLoopbackCapture: boolean;
	appAudioOnly: boolean;
	includeMicInput: boolean;
	ownSpeakerName: string;
	micDeviceContains: string;
	aiQualityMode: AiQualityMode;
	summaryModel: string;
	targetApps: string;
	expectedSpeakers: number;
	includeFullTranscript: boolean;
	autoSummarize: boolean;
	meetingLanguage: string;
}

const DEFAULT_SETTINGS: MeetingScribeSettings = {
	serverBaseUrl: 'http://127.0.0.1:8000',
	ollamaBaseUrl: 'http://127.0.0.1:11434',
	defaultSummaryModel: '',
	defaultFolder: '/',
	defaultMeetingLanguage: 'auto',
	defaultTargetApps: 'discord.exe,teams.exe,zoom.exe',
	defaultMicDeviceContains: '',
	defaultIncludeMicInput: true,
	defaultOwnSpeakerName: 'Me',
	defaultAiQualityMode: 'full_vram_12gb',
	autoSummarize: true,
	includeFullTranscript: true,
	expectedSpeakers: 3,
	captureProfile: 'balanced',
	enableDiarization: true,
	preferLoopbackCapture: true
};

const FALLBACK_LANGUAGE_OPTIONS: LanguageOption[] = [
	{ code: 'auto', label: 'Auto detect (recommended for mixed language / Denglisch)' },
	{ code: 'de', label: 'German (de)' },
	{ code: 'en', label: 'English (en)' }
];

function defaultMeetingFileName(): string {
	const now = new Date();
	const pad = (value: number): string => value.toString().padStart(2, '0');
	const yyyy = now.getFullYear();
	const mm = pad(now.getMonth() + 1);
	const dd = pad(now.getDate());
	const hh = pad(now.getHours());
	const min = pad(now.getMinutes());
	return `Meeting - ${yyyy}-${mm}-${dd} ${hh}-${min}`;
}

export default class MeetingScribePlugin extends Plugin {
	private ribbonIconEl!: HTMLElement;
	private isRecording = false;
	private settings: MeetingScribeSettings = DEFAULT_SETTINGS;
	private currentSessionOptions: SessionCaptureOptions | null = null;

	async onload(): Promise<void> {
		await this.loadSettings();
		this.addSettingTab(new MeetingScribeSettingTab(this.app, this));

		this.addCommand({
			id: 'meeting-scribe-toggle-capture',
			name: 'Start / Stop capture',
			callback: async () => this.toggleRecording()
		});

		this.addCommand({
			id: 'meeting-scribe-start-capture',
			name: 'Start capture',
			callback: async () => this.startRecording()
		});

		this.addCommand({
			id: 'meeting-scribe-stop-capture',
			name: 'Stop capture and generate notes',
			callback: async () => this.stopRecording()
		});

		this.addCommand({
			id: 'meeting-scribe-toggle-full-transcript',
			name: 'Toggle include full transcript in saved notes',
			callback: async () => {
				this.settings.includeFullTranscript = !this.settings.includeFullTranscript;
				await this.saveSettings();
				new Notice(
					this.settings.includeFullTranscript
						? 'Full transcript will be included in saved notes.'
						: 'Full transcript will be omitted from saved notes.'
				);
			}
		});

		this.ribbonIconEl = this.addRibbonIcon('mic', 'Start Meeting Scribe capture', async () => this.toggleRecording());
		this.updateRibbonUi();
	}

	private async loadSettings(): Promise<void> {
		this.settings = { ...DEFAULT_SETTINGS, ...(await this.loadData()) };
	}

	async saveSettings(): Promise<void> {
		await this.saveData(this.settings);
	}

	private updateRibbonUi(icon: 'mic' | 'square' | 'loader' = this.isRecording ? 'square' : 'mic'): void {
		if (!this.ribbonIconEl) {
			return;
		}
		setIcon(this.ribbonIconEl, icon);
		const label = this.isRecording ? 'Stop Meeting Scribe capture' : 'Start Meeting Scribe capture';
		this.ribbonIconEl.setAttribute('aria-label', label);
	}

	private async toggleRecording(): Promise<void> {
		if (this.isRecording) {
			await this.stopRecording();
			return;
		}
		await this.startRecording();
	}

	private async postJson<T>(path: string, payload?: unknown): Promise<T> {
		const url = `${this.settings.serverBaseUrl}${path}`;
		const response = await fetch(url, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: payload ? JSON.stringify(payload) : undefined
		});
		if (!response.ok) {
			throw new Error(`Server call failed (${response.status}) on ${path}`);
		}
		return (await response.json()) as T;
	}

	private async getJson<T>(path: string): Promise<T> {
		const url = `${this.settings.serverBaseUrl}${path}`;
		const response = await fetch(url, { method: 'GET' });
		if (!response.ok) {
			throw new Error(`Server call failed (${response.status}) on ${path}`);
		}
		return (await response.json()) as T;
	}

	async fetchMicDevices(): Promise<MicDevice[]> {
		const data = await this.getJson<{ devices: MicDevice[] }>('/devices/mics');
		return data.devices ?? [];
	}

	async fetchSupportedLanguages(): Promise<LanguageOption[]> {
		try {
			const data = await this.getJson<{ languages: LanguageOption[] }>('/languages');
			const languages = data.languages ?? [];
			return languages.length > 0 ? languages : FALLBACK_LANGUAGE_OPTIONS;
		} catch (error) {
			console.warn('Could not load language list, using fallback:', error);
			return FALLBACK_LANGUAGE_OPTIONS;
		}
	}

	async fetchLlmModels(ollamaBaseUrl: string): Promise<LlmModelOption[]> {
		const base = (ollamaBaseUrl || DEFAULT_SETTINGS.ollamaBaseUrl).trim() || DEFAULT_SETTINGS.ollamaBaseUrl;
		const encoded = encodeURIComponent(base);
		const data = await this.getJson<{ models: LlmModelOption[] }>(`/llm/models?ollama_url=${encoded}`);
		return data.models ?? [];
	}

	async testMicDevice(micDeviceContains: string): Promise<MicTestResponse> {
		return await this.postJson<MicTestResponse>('/test-mic', {
			mic_device_contains: micDeviceContains,
			duration_seconds: 1.8
		});
	}

	private async startRecording(): Promise<void> {
		if (this.isRecording) {
			new Notice('Capture already running.');
			return;
		}

		let micDevices: MicDevice[] = [];
		let languageOptions: LanguageOption[] = FALLBACK_LANGUAGE_OPTIONS;
		let llmModelOptions: LlmModelOption[] = [];
		try {
			const devicesResponse = await this.getJson<{ devices: MicDevice[] }>('/devices/mics');
			micDevices = devicesResponse.devices ?? [];
		} catch (error) {
			console.warn('Could not load microphone device list:', error);
		}
		try {
			languageOptions = await this.fetchSupportedLanguages();
		} catch (error) {
			console.warn('Could not load ASR language list:', error);
		}
		try {
			llmModelOptions = await this.fetchLlmModels(this.settings.ollamaBaseUrl);
		} catch (error) {
			console.warn('Could not load LLM model list:', error);
		}

		const sessionOptions = await new SessionConfigModal(
			this.app,
			this.settings,
			micDevices,
			languageOptions,
			llmModelOptions
		).openAndGetResult();
		if (!sessionOptions) {
			new Notice('Capture start cancelled.');
			return;
		}

		try {
			await this.postJson('/start', {
				profile: sessionOptions.captureProfile,
				enable_diarization: sessionOptions.enableDiarization,
				prefer_loopback: sessionOptions.preferLoopbackCapture,
				expected_speakers: sessionOptions.expectedSpeakers,
				language: sessionOptions.meetingLanguage,
				app_audio_only: sessionOptions.appAudioOnly,
				include_mic: sessionOptions.includeMicInput,
				self_speaker_name: sessionOptions.ownSpeakerName,
				mic_device_contains: sessionOptions.micDeviceContains,
				ai_quality_mode: sessionOptions.aiQualityMode,
				target_apps: sessionOptions.targetApps
					.split(',')
					.map((part) => part.trim().toLowerCase())
					.filter((part) => part.length > 0)
			});
			this.currentSessionOptions = sessionOptions;
			this.isRecording = true;
			this.updateRibbonUi();
			new Notice(`Recording started (${sessionOptions.captureProfile}, language: ${sessionOptions.meetingLanguage}).`);
		} catch (error) {
			console.error(error);
			new Notice('Unable to start. Is the Python companion server running?');
		}
	}

	private async stopRecording(): Promise<void> {
		if (!this.isRecording) {
			new Notice('No capture is currently running.');
			return;
		}

		this.isRecording = false;
		this.updateRibbonUi('loader');
		new Notice('Stopping capture and processing audio...');
		const sessionOptions = this.currentSessionOptions ?? {
			captureProfile: this.settings.captureProfile,
			enableDiarization: this.settings.enableDiarization,
			preferLoopbackCapture: this.settings.preferLoopbackCapture,
			appAudioOnly: true,
			includeMicInput: this.settings.defaultIncludeMicInput,
			ownSpeakerName: this.settings.defaultOwnSpeakerName,
			micDeviceContains: this.settings.defaultMicDeviceContains,
			aiQualityMode: this.settings.defaultAiQualityMode,
			targetApps: this.settings.defaultTargetApps,
			summaryModel: this.settings.defaultSummaryModel,
			expectedSpeakers: this.settings.expectedSpeakers,
			includeFullTranscript: this.settings.includeFullTranscript,
			autoSummarize: this.settings.autoSummarize,
			meetingLanguage: this.settings.defaultMeetingLanguage
		};

		try {
			const stopData = await this.postJson<StopResponse>('/stop');
			const speakerMap = await new SpeakerMapModal(
				this.app,
				stopData.speakers,
				stopData.samples,
				stopData.audio_samples ?? {},
				sessionOptions.ownSpeakerName
			).openAndGetResult();
			if (!speakerMap) {
				new Notice('Speaker labeling cancelled. Capture result not saved.');
				this.updateRibbonUi('mic');
				return;
			}

			const fullTranscript = this.buildTranscript(stopData.segments, speakerMap);
			let summary = '### Meeting Summary\n- Summary disabled in settings.';

			if (sessionOptions.autoSummarize) {
				new Notice('Generating local AI summary...');
				const sumData = await this.postJson<SummarizeResponse>('/summarize', {
					text: fullTranscript,
					language: sessionOptions.meetingLanguage,
					model: sessionOptions.summaryModel || undefined,
					ollama_url: this.settings.ollamaBaseUrl || undefined
				});
				summary = sumData.summary?.trim() || summary;
			}

			const saveTarget = await new SaveFileModal(this.app, this.settings.defaultFolder).openAndGetResult();
			if (!saveTarget) {
				new Notice('Save cancelled.');
				this.updateRibbonUi('mic');
				return;
			}

			const content = sessionOptions.includeFullTranscript
				? `${summary}\n\n---\n### Full Transcript\n\n${fullTranscript}`
				: summary;
			const filePath = await this.createMeetingFile(saveTarget.filename, saveTarget.folder, content);
			this.app.workspace.openLinkText(filePath, '', true);
			new Notice(`Meeting saved to ${filePath}`);
		} catch (error) {
			console.error(error);
			new Notice('Error processing meeting capture.');
		} finally {
			try {
				await this.postJson('/cleanup');
			} catch (cleanupError) {
				console.warn('Could not trigger backend GPU cleanup:', cleanupError);
			}
			this.currentSessionOptions = null;
			this.updateRibbonUi('mic');
		}
	}

	private buildTranscript(segments: Segment[], speakerMap: Map<string, string>): string {
		return segments
			.map((segment) => {
				const rawSpeaker = segment.speaker ?? 'UNKNOWN';
				const speakerName = speakerMap.get(rawSpeaker) || rawSpeaker;
				return `**${speakerName}:** ${segment.text.trim()}`;
			})
			.join('\n\n');
	}

	private async createMeetingFile(filename: string, folder: string, content: string): Promise<string> {
		const normalizedFolder = folder.trim() === '/' ? '' : folder.trim().replace(/^\/+|\/+$/g, '');
		if (normalizedFolder.length > 0) {
			await this.app.vault.createFolder(normalizedFolder).catch(() => {
				// Ignore error if folder already exists.
			});
		}

		const safeFileName = filename.trim().replace(/[\\/:*?"<>|]/g, '-');
		const basePath = normalizedFolder.length > 0 ? `${normalizedFolder}/${safeFileName}` : safeFileName;
		let finalPath = normalizePath(`${basePath}.md`);
		let index = 1;

		while (this.app.vault.getAbstractFileByPath(finalPath)) {
			finalPath = normalizePath(`${basePath} (${index}).md`);
			index += 1;
		}

		await this.app.vault.create(finalPath, content);
		return finalPath;
	}

	getSettings(): MeetingScribeSettings {
		return this.settings;
	}
}

class MeetingScribeSettingTab extends PluginSettingTab {
	private readonly plugin: MeetingScribePlugin;

	constructor(app: App, plugin: MeetingScribePlugin) {
		super(app, plugin);
		this.plugin = plugin;
	}

	display(): void {
		const { containerEl } = this;
		const settings = this.plugin.getSettings();
		containerEl.empty();
		containerEl.createEl('h2', { text: 'Meeting Scribe Settings' });

		new Setting(containerEl)
			.setName('Companion server URL')
			.setDesc('HTTP endpoint for the local Python companion app.')
			.addText((text) =>
				text
					.setPlaceholder('http://127.0.0.1:8000')
					.setValue(settings.serverBaseUrl)
					.onChange(async (value) => {
						settings.serverBaseUrl = value.trim() || DEFAULT_SETTINGS.serverBaseUrl;
						await this.plugin.saveSettings();
					})
			);

		new Setting(containerEl)
			.setName('Ollama URL')
			.setDesc('Local Ollama base URL used for LLM summarization, e.g. http://127.0.0.1:11434')
			.addText((text) =>
				text
					.setPlaceholder(DEFAULT_SETTINGS.ollamaBaseUrl)
					.setValue(settings.ollamaBaseUrl)
					.onChange(async (value) => {
						settings.ollamaBaseUrl = value.trim() || DEFAULT_SETTINGS.ollamaBaseUrl;
						await this.plugin.saveSettings();
					})
			);

		this.renderLlmControls(containerEl, settings);

		new Setting(containerEl)
			.setName('Default save folder')
			.setDesc('Vault-relative folder used as default in the save prompt.')
			.addText((text) =>
				text.setPlaceholder('/').setValue(settings.defaultFolder).onChange(async (value) => {
					settings.defaultFolder = value.trim() || '/';
					await this.plugin.saveSettings();
				})
			);

		this.renderLanguageControls(containerEl, settings);

		new Setting(containerEl)
			.setName('Default target meeting apps')
			.setDesc('Comma-separated executables for app-audio filter, e.g. discord.exe,teams.exe,zoom.exe')
			.addText((text) =>
				text.setValue(settings.defaultTargetApps).onChange(async (value) => {
					settings.defaultTargetApps = value.trim() || DEFAULT_SETTINGS.defaultTargetApps;
					await this.plugin.saveSettings();
				})
			);

		new Setting(containerEl)
			.setName('Default include own microphone')
			.setDesc('If enabled, your mic is recorded in addition to meeting app audio.')
			.addToggle((toggle) =>
				toggle.setValue(settings.defaultIncludeMicInput).onChange(async (value) => {
					settings.defaultIncludeMicInput = value;
					await this.plugin.saveSettings();
				})
			);

		new Setting(containerEl)
			.setName('Default own speaker name')
			.setDesc('Used to prelabel your mic-detected speech as SELF_USER.')
			.addText((text) =>
				text.setValue(settings.defaultOwnSpeakerName).onChange(async (value) => {
					settings.defaultOwnSpeakerName = value.trim() || DEFAULT_SETTINGS.defaultOwnSpeakerName;
					await this.plugin.saveSettings();
				})
			);

		new Setting(containerEl)
			.setName('AI quality mode')
			.setDesc('Choose efficient defaults or a stronger 12GB VRAM mode for ASR/diarization quality.')
			.addDropdown((dropdown) => {
				dropdown.addOption('efficient', 'Efficient (lower VRAM)');
				dropdown.addOption('full_vram_12gb', 'Full VRAM 12GB (best quality)');
				dropdown.setValue(settings.defaultAiQualityMode);
				dropdown.onChange(async (value) => {
					if (value === 'efficient' || value === 'full_vram_12gb') {
						settings.defaultAiQualityMode = value;
						await this.plugin.saveSettings();
					}
				});
			});
		this.renderMicDeviceControls(containerEl, settings);

		new Setting(containerEl)
			.setName('Auto summarize')
			.setDesc('Call the local summarization endpoint after transcription.')
			.addToggle((toggle) =>
				toggle.setValue(settings.autoSummarize).onChange(async (value) => {
					settings.autoSummarize = value;
					await this.plugin.saveSettings();
				})
			);

		new Setting(containerEl)
			.setName('Include full transcript in saved note')
			.setDesc('If disabled, only the structured summary is saved.')
			.addToggle((toggle) =>
				toggle.setValue(settings.includeFullTranscript).onChange(async (value) => {
					settings.includeFullTranscript = value;
					await this.plugin.saveSettings();
				})
			);

		new Setting(containerEl)
			.setName('Capture profile')
			.setDesc('Fast = lower latency, Balanced = default, Pristine = highest quality.')
			.addDropdown((dropdown) => {
				dropdown.addOption('fast', 'Fast');
				dropdown.addOption('balanced', 'Balanced');
				dropdown.addOption('pristine', 'Pristine');
				dropdown.setValue(settings.captureProfile);
				dropdown.onChange(async (value) => {
					if (value === 'fast' || value === 'balanced' || value === 'pristine') {
						settings.captureProfile = value;
					}
					await this.plugin.saveSettings();
				});
			});

		new Setting(containerEl)
			.setName('Enable diarization')
			.setDesc('Use local speaker diarization with multiple backends and fallback clustering.')
			.addToggle((toggle) =>
				toggle.setValue(settings.enableDiarization).onChange(async (value) => {
					settings.enableDiarization = value;
					await this.plugin.saveSettings();
				})
			);

		new Setting(containerEl)
			.setName('Expected speaker count')
			.setDesc('Guides diarization clustering (1-8). Use 3 for most meetings with several participants.')
			.addSlider((slider) => {
				slider.setLimits(1, 8, 1);
				slider.setValue(settings.expectedSpeakers);
				slider.setDynamicTooltip();
				slider.onChange(async (value) => {
					settings.expectedSpeakers = value;
					await this.plugin.saveSettings();
				});
			});

		new Setting(containerEl)
			.setName('Prefer loopback + mic capture')
			.setDesc('On Windows, try WASAPI loopback + microphone mixing first.')
			.addToggle((toggle) =>
				toggle.setValue(settings.preferLoopbackCapture).onChange(async (value) => {
					settings.preferLoopbackCapture = value;
					await this.plugin.saveSettings();
				})
			);
	}

	private renderMicDeviceControls(containerEl: HTMLElement, settings: MeetingScribeSettings): void {
		const section = containerEl.createDiv();
		section.createEl('p', { text: 'Loading microphone devices...' });

		void (async () => {
			try {
				const devices = await this.plugin.fetchMicDevices();
				section.empty();

				new Setting(section)
					.setName('Default microphone device')
					.setDesc('This microphone is preselected as default for new sessions.')
					.addDropdown((dropdown) => {
						dropdown.addOption('', 'Default microphone');
						for (const mic of devices) {
							dropdown.addOption(mic.id, mic.name);
						}
						dropdown.setValue(settings.defaultMicDeviceContains || '');
						dropdown.onChange(async (value) => {
							settings.defaultMicDeviceContains = value;
							await this.plugin.saveSettings();
						});
					});

				new Setting(section)
					.setName('Microphone test')
					.setDesc('Runs a 2-second local signal check for the selected default mic.')
					.addButton((btn) =>
						btn.setButtonText('Test microphone').onClick(async () => {
							btn.setDisabled(true);
							btn.setButtonText('Testing...');
							try {
								const result = await this.plugin.testMicDevice(settings.defaultMicDeviceContains || '');
								if (!result.ok) {
									new Notice(`Mic test failed: ${result.error || 'Unknown error'}`);
								} else {
									new Notice(
										`Mic OK (${result.device_name}) - RMS ${result.rms.toFixed(1)}, Peak ${result.peak.toFixed(1)}`
									);
								}
							} catch (error) {
								console.error(error);
								new Notice('Mic test failed: Companion server not reachable.');
							} finally {
								btn.setDisabled(false);
								btn.setButtonText('Test microphone');
							}
						})
					);

				new Setting(section)
					.setName('Fallback mic filter (manual)')
					.setDesc('Used if dropdown detection is incomplete; set partial device name.')
					.addText((text) =>
						text
							.setPlaceholder('e.g. rode, shure, headset')
							.setValue(settings.defaultMicDeviceContains)
							.onChange(async (value) => {
								settings.defaultMicDeviceContains = value.trim();
								await this.plugin.saveSettings();
							})
					);
			} catch (error) {
				console.error(error);
				section.empty();
				section.createEl('p', { text: 'Could not load microphone devices. Companion server may be offline.' });
			}
		})();
	}

	private renderLlmControls(containerEl: HTMLElement, settings: MeetingScribeSettings): void {
		const section = containerEl.createDiv();
		section.createEl('p', { text: 'Loading Ollama models...' });
		void (async () => {
			try {
				const models = await this.plugin.fetchLlmModels(settings.ollamaBaseUrl);
				section.empty();
				new Setting(section)
					.setName('Summary model (LLM)')
					.setDesc('Model used for local LLM tasks (currently meeting summarization).')
					.addDropdown((dropdown) => {
						dropdown.addOption('', 'Structured fallback only (no LLM)');
						for (const model of models) {
							dropdown.addOption(model.id, model.name);
						}
						dropdown.setValue(settings.defaultSummaryModel || '');
						dropdown.onChange(async (value) => {
							settings.defaultSummaryModel = value;
							await this.plugin.saveSettings();
						});
					})
					.addButton((btn) =>
						btn.setButtonText('Reload models').onClick(async () => {
							this.display();
						})
					);
			} catch (error) {
				console.error(error);
				section.empty();
				section.createEl('p', {
					text: 'Could not load Ollama models. Check Ollama URL and that Ollama is running.'
				});
			}
		})();
	}

	private renderLanguageControls(containerEl: HTMLElement, settings: MeetingScribeSettings): void {
		const section = containerEl.createDiv();
		section.createEl('p', { text: 'Loading supported transcription languages...' });
		void (async () => {
			try {
				const languages = await this.plugin.fetchSupportedLanguages();
				section.empty();
				new Setting(section)
					.setName('Default meeting language')
					.setDesc('Used as prefilled value in the session dialog. Auto works best for Denglisch/code-switching.')
					.addDropdown((dropdown) => {
						for (const language of languages) {
							dropdown.addOption(language.code, language.label);
						}
						const knownCodes = new Set(languages.map((item) => item.code));
						const selected = knownCodes.has(settings.defaultMeetingLanguage)
							? settings.defaultMeetingLanguage
							: 'auto';
						dropdown.setValue(selected);
						dropdown.onChange(async (value) => {
							settings.defaultMeetingLanguage = value || 'auto';
							await this.plugin.saveSettings();
						});
					});
			} catch (error) {
				console.error(error);
				section.empty();
				section.createEl('p', { text: 'Could not load language list. Falling back to auto detection.' });
			}
		})();
	}
}

class SpeakerMapModal extends Modal {
	private readonly speakers: string[];
	private readonly samples: Record<string, string>;
	private readonly audioSamples: Record<string, string>;
	private readonly ownSpeakerName: string;
	private readonly speakerMap = new Map<string, string>();
	private resolver!: (result: Map<string, string> | null) => void;
	private submitted = false;

	constructor(
		app: App,
		speakers: string[],
		samples: Record<string, string>,
		audioSamples: Record<string, string>,
		ownSpeakerName: string
	) {
		super(app);
		this.speakers = speakers;
		this.samples = samples;
		this.audioSamples = audioSamples;
		this.ownSpeakerName = ownSpeakerName?.trim() || 'Me';
	}

	openAndGetResult(): Promise<Map<string, string> | null> {
		this.open();
		return new Promise((resolve) => {
			this.resolver = resolve;
		});
	}

	onOpen(): void {
		const { contentEl } = this;
		contentEl.empty();
		contentEl.createEl('h2', { text: 'Identify Speakers' });
		contentEl.createEl('p', { text: 'Rename each detected speaker before saving.' });

		if (this.speakers.length === 0) {
			contentEl.createEl('p', { text: 'No speaker diarization labels were detected.' });
		}

		this.speakers.forEach((speakerId) => {
			const defaultLabel = speakerId === 'SELF_USER' ? this.ownSpeakerName : speakerId;
			this.speakerMap.set(speakerId, defaultLabel);

			const row = contentEl.createDiv({ cls: 'meeting-scribe-speaker-row' });
			row.style.marginBottom = '14px';
			const sampleText = this.samples[speakerId] || 'No sample available.';
			row.createEl('small', { text: `"${sampleText}"` }).style.fontStyle = 'italic';
			const audioBase64 = this.audioSamples[speakerId];
			if (audioBase64) {
				const audioEl = row.createEl('audio');
				audioEl.controls = true;
				audioEl.preload = 'none';
				audioEl.src = `data:audio/wav;base64,${audioBase64}`;
				audioEl.style.display = 'block';
				audioEl.style.marginTop = '8px';
			}

			new Setting(row)
				.setName(speakerId)
				.setDesc('Enter the real name')
				.addText((text) =>
					text
						.setPlaceholder('Speaker name')
						.setValue(defaultLabel)
						.onChange((value) => this.speakerMap.set(speakerId, value.trim() || speakerId))
				);
		});

		new Setting(contentEl)
			.addButton((btn) =>
				btn.setButtonText('Cancel').onClick(() => {
					this.submitted = false;
					this.close();
				})
			)
			.addButton((btn) =>
				btn
					.setButtonText('Continue')
					.setCta()
					.onClick(() => {
						this.submitted = true;
						this.close();
					})
			);
	}

	onClose(): void {
		this.contentEl.empty();
		if (this.resolver) {
			this.resolver(this.submitted ? this.speakerMap : null);
		}
	}
}

class SaveFileModal extends Modal {
	private filename = defaultMeetingFileName();
	private folder: string;
	private resolver!: (result: { filename: string; folder: string } | null) => void;
	private submitted = false;

	constructor(app: App, defaultFolder: string) {
		super(app);
		this.folder = defaultFolder || '/';
	}

	openAndGetResult(): Promise<{ filename: string; folder: string } | null> {
		this.open();
		return new Promise((resolve) => {
			this.resolver = resolve;
		});
	}

	onOpen(): void {
		const { contentEl } = this;
		contentEl.empty();
		contentEl.createEl('h2', { text: 'Save Meeting Notes' });

		new Setting(contentEl)
			.setName('Filename')
			.addText((text) => text.setValue(this.filename).onChange((val) => (this.filename = val)));

		new Setting(contentEl)
			.setName('Folder Path')
			.setDesc('Example: Meetings/Work (use / for vault root)')
			.addText((text) => text.setValue(this.folder).onChange((val) => (this.folder = val)));

		const knownFolders = this.app.vault
			.getAllLoadedFiles()
			.filter((file): file is TFolder => file instanceof TFolder)
			.map((folder) => folder.path)
			.filter((path) => path.length > 0)
			.sort();

		if (knownFolders.length > 0) {
			new Setting(contentEl)
				.setName('Quick pick existing folder')
				.addDropdown((dropdown) => {
					dropdown.addOption('/', '/ (vault root)');
					for (const folderPath of knownFolders) {
						dropdown.addOption(folderPath, folderPath);
					}
					dropdown.setValue(this.folder);
					dropdown.onChange((value) => {
						this.folder = value;
					});
				});
		}

		new Setting(contentEl)
			.addButton((btn) =>
				btn.setButtonText('Cancel').onClick(() => {
					this.submitted = false;
					this.close();
				})
			)
			.addButton((btn) =>
				btn
					.setButtonText('Save')
					.setCta()
					.onClick(() => {
						this.submitted = true;
						this.close();
					})
			);
	}

	onClose(): void {
		this.contentEl.empty();
		if (!this.resolver) {
			return;
		}
		if (!this.submitted) {
			this.resolver(null);
			return;
		}
		this.resolver({
			filename: this.filename.trim() || defaultMeetingFileName(),
			folder: this.folder.trim() || '/'
		});
	}
}

class SessionConfigModal extends Modal {
	private readonly baseSettings: MeetingScribeSettings;
	private readonly micDevices: MicDevice[];
	private readonly languageOptions: LanguageOption[];
	private readonly llmModelOptions: LlmModelOption[];
	private options: SessionCaptureOptions;
	private resolver!: (result: SessionCaptureOptions | null) => void;
	private submitted = false;

	constructor(
		app: App,
		settings: MeetingScribeSettings,
		micDevices: MicDevice[],
		languageOptions: LanguageOption[],
		llmModelOptions: LlmModelOption[]
	) {
		super(app);
		this.baseSettings = settings;
		this.micDevices = micDevices;
		this.languageOptions = languageOptions.length > 0 ? languageOptions : FALLBACK_LANGUAGE_OPTIONS;
		this.llmModelOptions = llmModelOptions;
		this.options = {
			captureProfile: settings.captureProfile,
			enableDiarization: settings.enableDiarization,
			preferLoopbackCapture: settings.preferLoopbackCapture,
			appAudioOnly: true,
			includeMicInput: settings.defaultIncludeMicInput,
			ownSpeakerName: settings.defaultOwnSpeakerName,
			micDeviceContains: settings.defaultMicDeviceContains,
			aiQualityMode: settings.defaultAiQualityMode,
			summaryModel: settings.defaultSummaryModel,
			targetApps: settings.defaultTargetApps,
			expectedSpeakers: settings.expectedSpeakers,
			includeFullTranscript: settings.includeFullTranscript,
			autoSummarize: settings.autoSummarize,
			meetingLanguage: settings.defaultMeetingLanguage || 'auto'
		};
	}

	openAndGetResult(): Promise<SessionCaptureOptions | null> {
		this.open();
		return new Promise((resolve) => {
			this.resolver = resolve;
		});
	}

	onOpen(): void {
		const { contentEl } = this;
		contentEl.empty();
		contentEl.createEl('h2', { text: 'Session Settings' });
		contentEl.createEl('p', { text: 'Configure this capture session before recording starts.' });

		new Setting(contentEl)
			.setName('Meeting language')
			.setDesc('Select forced transcription language, or Auto for mixed-language speech (recommended for Denglisch).')
			.addDropdown((dropdown) => {
				for (const language of this.languageOptions) {
					dropdown.addOption(language.code, language.label);
				}
				const knownCodes = new Set(this.languageOptions.map((item) => item.code));
				const selected = knownCodes.has(this.options.meetingLanguage) ? this.options.meetingLanguage : 'auto';
				dropdown.setValue(selected);
				dropdown.onChange((value) => {
					this.options.meetingLanguage = value || 'auto';
				});
			});

		new Setting(contentEl)
			.setName('Expected participants')
			.setDesc('Helps diarization separate speakers better.')
			.addSlider((slider) => {
				slider.setLimits(1, 8, 1);
				slider.setValue(this.options.expectedSpeakers);
				slider.setDynamicTooltip();
				slider.onChange((value) => {
					this.options.expectedSpeakers = value;
				});
			});

		new Setting(contentEl)
			.setName('Capture profile')
			.setDesc('Fast = lower latency, Balanced = default, Pristine = highest quality.')
			.addDropdown((dropdown) => {
				dropdown.addOption('fast', 'Fast');
				dropdown.addOption('balanced', 'Balanced');
				dropdown.addOption('pristine', 'Pristine');
				dropdown.setValue(this.options.captureProfile);
				dropdown.onChange((value) => {
					if (value === 'fast' || value === 'balanced' || value === 'pristine') {
						this.options.captureProfile = value;
					}
				});
			});

		new Setting(contentEl)
			.setName('AI quality mode')
			.setDesc('Use full 12GB mode for stronger local ASR and speaker separation.')
			.addDropdown((dropdown) => {
				dropdown.addOption('efficient', 'Efficient (lower VRAM)');
				dropdown.addOption('full_vram_12gb', 'Full VRAM 12GB (best quality)');
				dropdown.setValue(this.options.aiQualityMode);
				dropdown.onChange((value) => {
					if (value === 'efficient' || value === 'full_vram_12gb') {
						this.options.aiQualityMode = value;
					}
				});
			});

		new Setting(contentEl)
			.setName('Enable diarization')
			.addToggle((toggle) =>
				toggle.setValue(this.options.enableDiarization).onChange((value) => {
					this.options.enableDiarization = value;
				})
			);

		new Setting(contentEl)
			.setName('Prefer loopback + mic capture')
			.addToggle((toggle) =>
				toggle.setValue(this.options.preferLoopbackCapture).onChange((value) => {
					this.options.preferLoopbackCapture = value;
				})
			);

		new Setting(contentEl)
			.setName('Meeting apps only (system audio filter)')
			.setDesc('Use only system audio when selected meeting apps are actively outputting sound.')
			.addToggle((toggle) =>
				toggle.setValue(this.options.appAudioOnly).onChange((value) => {
					this.options.appAudioOnly = value;
				})
			);

		new Setting(contentEl)
			.setName('Meeting app executables')
			.setDesc('Comma-separated, e.g. discord.exe, teams.exe, zoom.exe')
			.addText((text) =>
				text
					.setValue(this.options.targetApps)
					.onChange((value) => (this.options.targetApps = value.trim()))
			);

		new Setting(contentEl)
			.setName('Include own microphone input')
			.setDesc('If enabled, your mic is included in the recording even with meeting-app filter enabled.')
			.addToggle((toggle) =>
				toggle.setValue(this.options.includeMicInput).onChange((value) => {
					this.options.includeMicInput = value;
				})
			);

		new Setting(contentEl)
			.setName('Own speaker name')
			.setDesc('Prelabel mic-dominant speech as this name.')
			.addText((text) =>
				text
					.setValue(this.options.ownSpeakerName)
					.onChange((value) => (this.options.ownSpeakerName = value.trim() || 'Me'))
			);

		if (this.micDevices.length > 0) {
			new Setting(contentEl)
				.setName('Microphone device')
				.setDesc('Select the microphone to use for your own speech.')
				.addDropdown((dropdown) => {
					dropdown.addOption('', 'Default microphone');
					for (const mic of this.micDevices) {
						dropdown.addOption(mic.id, mic.name);
					}
					dropdown.setValue(this.options.micDeviceContains || '');
					dropdown.onChange((value) => {
						this.options.micDeviceContains = value;
					});
				});
		}
		else {
			new Setting(contentEl)
				.setName('Microphone device filter')
				.setDesc('Could not load mic list; use partial device name.')
				.addText((text) =>
					text.setValue(this.options.micDeviceContains).onChange((value) => (this.options.micDeviceContains = value.trim()))
				);
		}

		new Setting(contentEl)
			.setName('Auto summarize')
			.addToggle((toggle) =>
				toggle.setValue(this.options.autoSummarize).onChange((value) => {
					this.options.autoSummarize = value;
				})
			);

		new Setting(contentEl)
			.setName('Summary model for this session')
			.setDesc('Overrides the default LLM model only for this capture session.')
			.addDropdown((dropdown) => {
				dropdown.addOption('', 'Use default summary model');
				for (const model of this.llmModelOptions) {
					dropdown.addOption(model.id, model.name);
				}
				if (
					this.options.summaryModel &&
					!this.llmModelOptions.some((model) => model.id === this.options.summaryModel)
				) {
					dropdown.addOption(this.options.summaryModel, `${this.options.summaryModel} (custom)`);
				}
				dropdown.setValue(this.options.summaryModel || '');
				dropdown.onChange((value) => {
					this.options.summaryModel = value;
				});
			});

		new Setting(contentEl)
			.setName('Include full transcript in final note')
			.addToggle((toggle) =>
				toggle.setValue(this.options.includeFullTranscript).onChange((value) => {
					this.options.includeFullTranscript = value;
				})
			);

		new Setting(contentEl)
			.addButton((btn) =>
				btn.setButtonText('Cancel').onClick(() => {
					this.submitted = false;
					this.close();
				})
			)
			.addButton((btn) =>
				btn
					.setButtonText('Start Capture')
					.setCta()
					.onClick(() => {
						this.submitted = true;
						this.close();
					})
			);
	}

	onClose(): void {
		this.contentEl.empty();
		if (!this.resolver) {
			return;
		}
		this.resolver(this.submitted ? this.options : null);
	}
}