"use strict";
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, { get: all[name], enumerable: true });
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function") {
    for (let key of __getOwnPropNames(from))
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
  }
  return to;
};
var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

// main.ts
var main_exports = {};
__export(main_exports, {
  default: () => MeetingScribePlugin
});
module.exports = __toCommonJS(main_exports);
var import_obsidian = require("obsidian");
var DEFAULT_SETTINGS = {
  serverBaseUrl: "http://127.0.0.1:8000",
  ollamaBaseUrl: "http://127.0.0.1:11434",
  defaultSummaryModel: "",
  defaultFolder: "/",
  defaultMeetingLanguage: "auto",
  defaultTargetApps: "discord.exe,teams.exe,zoom.exe",
  defaultMicDeviceContains: "",
  defaultIncludeMicInput: true,
  defaultOwnSpeakerName: "Me",
  defaultAiQualityMode: "full_vram_12gb",
  autoSummarize: true,
  includeFullTranscript: true,
  expectedSpeakers: 3,
  captureProfile: "balanced",
  enableDiarization: true,
  preferLoopbackCapture: true
};
var FALLBACK_LANGUAGE_OPTIONS = [
  { code: "auto", label: "Auto detect (recommended for mixed language / Denglisch)" },
  { code: "de", label: "German (de)" },
  { code: "en", label: "English (en)" }
];
function defaultMeetingFileName() {
  const now = /* @__PURE__ */ new Date();
  const pad = (value) => value.toString().padStart(2, "0");
  const yyyy = now.getFullYear();
  const mm = pad(now.getMonth() + 1);
  const dd = pad(now.getDate());
  const hh = pad(now.getHours());
  const min = pad(now.getMinutes());
  return `Meeting - ${yyyy}-${mm}-${dd} ${hh}-${min}`;
}
var MeetingScribePlugin = class extends import_obsidian.Plugin {
  constructor() {
    super(...arguments);
    this.isRecording = false;
    this.settings = DEFAULT_SETTINGS;
    this.currentSessionOptions = null;
  }
  async onload() {
    await this.loadSettings();
    this.addSettingTab(new MeetingScribeSettingTab(this.app, this));
    this.addCommand({
      id: "meeting-scribe-toggle-capture",
      name: "Start / Stop capture",
      callback: async () => this.toggleRecording()
    });
    this.addCommand({
      id: "meeting-scribe-start-capture",
      name: "Start capture",
      callback: async () => this.startRecording()
    });
    this.addCommand({
      id: "meeting-scribe-stop-capture",
      name: "Stop capture and generate notes",
      callback: async () => this.stopRecording()
    });
    this.addCommand({
      id: "meeting-scribe-toggle-full-transcript",
      name: "Toggle include full transcript in saved notes",
      callback: async () => {
        this.settings.includeFullTranscript = !this.settings.includeFullTranscript;
        await this.saveSettings();
        new import_obsidian.Notice(
          this.settings.includeFullTranscript ? "Full transcript will be included in saved notes." : "Full transcript will be omitted from saved notes."
        );
      }
    });
    this.ribbonIconEl = this.addRibbonIcon("mic", "Start Meeting Scribe capture", async () => this.toggleRecording());
    this.updateRibbonUi();
  }
  async loadSettings() {
    this.settings = { ...DEFAULT_SETTINGS, ...await this.loadData() };
  }
  async saveSettings() {
    await this.saveData(this.settings);
  }
  updateRibbonUi(icon = this.isRecording ? "square" : "mic") {
    if (!this.ribbonIconEl) {
      return;
    }
    (0, import_obsidian.setIcon)(this.ribbonIconEl, icon);
    const label = this.isRecording ? "Stop Meeting Scribe capture" : "Start Meeting Scribe capture";
    this.ribbonIconEl.setAttribute("aria-label", label);
  }
  async toggleRecording() {
    if (this.isRecording) {
      await this.stopRecording();
      return;
    }
    await this.startRecording();
  }
  async postJson(path, payload) {
    const url = `${this.settings.serverBaseUrl}${path}`;
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload ? JSON.stringify(payload) : void 0
    });
    if (!response.ok) {
      throw new Error(`Server call failed (${response.status}) on ${path}`);
    }
    return await response.json();
  }
  async getJson(path) {
    const url = `${this.settings.serverBaseUrl}${path}`;
    const response = await fetch(url, { method: "GET" });
    if (!response.ok) {
      throw new Error(`Server call failed (${response.status}) on ${path}`);
    }
    return await response.json();
  }
  async fetchMicDevices() {
    const data = await this.getJson("/devices/mics");
    return data.devices ?? [];
  }
  async fetchSupportedLanguages() {
    try {
      const data = await this.getJson("/languages");
      const languages = data.languages ?? [];
      return languages.length > 0 ? languages : FALLBACK_LANGUAGE_OPTIONS;
    } catch (error) {
      console.warn("Could not load language list, using fallback:", error);
      return FALLBACK_LANGUAGE_OPTIONS;
    }
  }
  async fetchLlmModels(ollamaBaseUrl) {
    const base = (ollamaBaseUrl || DEFAULT_SETTINGS.ollamaBaseUrl).trim() || DEFAULT_SETTINGS.ollamaBaseUrl;
    const encoded = encodeURIComponent(base);
    const data = await this.getJson(`/llm/models?ollama_url=${encoded}`);
    return data.models ?? [];
  }
  async testMicDevice(micDeviceContains) {
    return await this.postJson("/test-mic", {
      mic_device_contains: micDeviceContains,
      duration_seconds: 1.8
    });
  }
  async startRecording() {
    if (this.isRecording) {
      new import_obsidian.Notice("Capture already running.");
      return;
    }
    let micDevices = [];
    let languageOptions = FALLBACK_LANGUAGE_OPTIONS;
    let llmModelOptions = [];
    try {
      const devicesResponse = await this.getJson("/devices/mics");
      micDevices = devicesResponse.devices ?? [];
    } catch (error) {
      console.warn("Could not load microphone device list:", error);
    }
    try {
      languageOptions = await this.fetchSupportedLanguages();
    } catch (error) {
      console.warn("Could not load ASR language list:", error);
    }
    try {
      llmModelOptions = await this.fetchLlmModels(this.settings.ollamaBaseUrl);
    } catch (error) {
      console.warn("Could not load LLM model list:", error);
    }
    const sessionOptions = await new SessionConfigModal(
      this.app,
      this.settings,
      micDevices,
      languageOptions,
      llmModelOptions
    ).openAndGetResult();
    if (!sessionOptions) {
      new import_obsidian.Notice("Capture start cancelled.");
      return;
    }
    try {
      await this.postJson("/start", {
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
        target_apps: sessionOptions.targetApps.split(",").map((part) => part.trim().toLowerCase()).filter((part) => part.length > 0)
      });
      this.currentSessionOptions = sessionOptions;
      this.isRecording = true;
      this.updateRibbonUi();
      new import_obsidian.Notice(`Recording started (${sessionOptions.captureProfile}, language: ${sessionOptions.meetingLanguage}).`);
    } catch (error) {
      console.error(error);
      new import_obsidian.Notice("Unable to start. Is the Python companion server running?");
    }
  }
  async stopRecording() {
    if (!this.isRecording) {
      new import_obsidian.Notice("No capture is currently running.");
      return;
    }
    this.isRecording = false;
    this.updateRibbonUi("loader");
    new import_obsidian.Notice("Stopping capture and processing audio...");
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
      const stopData = await this.postJson("/stop");
      const speakerMap = await new SpeakerMapModal(
        this.app,
        stopData.speakers,
        stopData.samples,
        stopData.audio_samples ?? {},
        sessionOptions.ownSpeakerName
      ).openAndGetResult();
      if (!speakerMap) {
        new import_obsidian.Notice("Speaker labeling cancelled. Capture result not saved.");
        this.updateRibbonUi("mic");
        return;
      }
      const fullTranscript = this.buildTranscript(stopData.segments, speakerMap);
      let summary = "### Meeting Summary\n- Summary disabled in settings.";
      if (sessionOptions.autoSummarize) {
        new import_obsidian.Notice("Generating local AI summary...");
        const sumData = await this.postJson("/summarize", {
          text: fullTranscript,
          language: sessionOptions.meetingLanguage,
          model: sessionOptions.summaryModel || void 0,
          ollama_url: this.settings.ollamaBaseUrl || void 0
        });
        summary = sumData.summary?.trim() || summary;
      }
      const saveTarget = await new SaveFileModal(this.app, this.settings.defaultFolder).openAndGetResult();
      if (!saveTarget) {
        new import_obsidian.Notice("Save cancelled.");
        this.updateRibbonUi("mic");
        return;
      }
      const content = sessionOptions.includeFullTranscript ? `${summary}

---
### Full Transcript

${fullTranscript}` : summary;
      const filePath = await this.createMeetingFile(saveTarget.filename, saveTarget.folder, content);
      this.app.workspace.openLinkText(filePath, "", true);
      new import_obsidian.Notice(`Meeting saved to ${filePath}`);
    } catch (error) {
      console.error(error);
      new import_obsidian.Notice("Error processing meeting capture.");
    } finally {
      try {
        await this.postJson("/cleanup");
      } catch (cleanupError) {
        console.warn("Could not trigger backend GPU cleanup:", cleanupError);
      }
      this.currentSessionOptions = null;
      this.updateRibbonUi("mic");
    }
  }
  buildTranscript(segments, speakerMap) {
    return segments.map((segment) => {
      const rawSpeaker = segment.speaker ?? "UNKNOWN";
      const speakerName = speakerMap.get(rawSpeaker) || rawSpeaker;
      return `**${speakerName}:** ${segment.text.trim()}`;
    }).join("\n\n");
  }
  async createMeetingFile(filename, folder, content) {
    const normalizedFolder = folder.trim() === "/" ? "" : folder.trim().replace(/^\/+|\/+$/g, "");
    if (normalizedFolder.length > 0) {
      await this.app.vault.createFolder(normalizedFolder).catch(() => {
      });
    }
    const safeFileName = filename.trim().replace(/[\\/:*?"<>|]/g, "-");
    const basePath = normalizedFolder.length > 0 ? `${normalizedFolder}/${safeFileName}` : safeFileName;
    let finalPath = (0, import_obsidian.normalizePath)(`${basePath}.md`);
    let index = 1;
    while (this.app.vault.getAbstractFileByPath(finalPath)) {
      finalPath = (0, import_obsidian.normalizePath)(`${basePath} (${index}).md`);
      index += 1;
    }
    await this.app.vault.create(finalPath, content);
    return finalPath;
  }
  getSettings() {
    return this.settings;
  }
};
var MeetingScribeSettingTab = class extends import_obsidian.PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }
  display() {
    const { containerEl } = this;
    const settings = this.plugin.getSettings();
    containerEl.empty();
    containerEl.createEl("h2", { text: "Meeting Scribe Settings" });
    new import_obsidian.Setting(containerEl).setName("Companion server URL").setDesc("HTTP endpoint for the local Python companion app.").addText(
      (text) => text.setPlaceholder("http://127.0.0.1:8000").setValue(settings.serverBaseUrl).onChange(async (value) => {
        settings.serverBaseUrl = value.trim() || DEFAULT_SETTINGS.serverBaseUrl;
        await this.plugin.saveSettings();
      })
    );
    new import_obsidian.Setting(containerEl).setName("Ollama URL").setDesc("Local Ollama base URL used for LLM summarization, e.g. http://127.0.0.1:11434").addText(
      (text) => text.setPlaceholder(DEFAULT_SETTINGS.ollamaBaseUrl).setValue(settings.ollamaBaseUrl).onChange(async (value) => {
        settings.ollamaBaseUrl = value.trim() || DEFAULT_SETTINGS.ollamaBaseUrl;
        await this.plugin.saveSettings();
      })
    );
    this.renderLlmControls(containerEl, settings);
    new import_obsidian.Setting(containerEl).setName("Default save folder").setDesc("Vault-relative folder used as default in the save prompt.").addText(
      (text) => text.setPlaceholder("/").setValue(settings.defaultFolder).onChange(async (value) => {
        settings.defaultFolder = value.trim() || "/";
        await this.plugin.saveSettings();
      })
    );
    this.renderLanguageControls(containerEl, settings);
    new import_obsidian.Setting(containerEl).setName("Default target meeting apps").setDesc("Comma-separated executables for app-audio filter, e.g. discord.exe,teams.exe,zoom.exe").addText(
      (text) => text.setValue(settings.defaultTargetApps).onChange(async (value) => {
        settings.defaultTargetApps = value.trim() || DEFAULT_SETTINGS.defaultTargetApps;
        await this.plugin.saveSettings();
      })
    );
    new import_obsidian.Setting(containerEl).setName("Default include own microphone").setDesc("If enabled, your mic is recorded in addition to meeting app audio.").addToggle(
      (toggle) => toggle.setValue(settings.defaultIncludeMicInput).onChange(async (value) => {
        settings.defaultIncludeMicInput = value;
        await this.plugin.saveSettings();
      })
    );
    new import_obsidian.Setting(containerEl).setName("Default own speaker name").setDesc("Used to prelabel your mic-detected speech as SELF_USER.").addText(
      (text) => text.setValue(settings.defaultOwnSpeakerName).onChange(async (value) => {
        settings.defaultOwnSpeakerName = value.trim() || DEFAULT_SETTINGS.defaultOwnSpeakerName;
        await this.plugin.saveSettings();
      })
    );
    new import_obsidian.Setting(containerEl).setName("AI quality mode").setDesc("Choose efficient defaults or a stronger 12GB VRAM mode for ASR/diarization quality.").addDropdown((dropdown) => {
      dropdown.addOption("efficient", "Efficient (lower VRAM)");
      dropdown.addOption("full_vram_12gb", "Full VRAM 12GB (best quality)");
      dropdown.setValue(settings.defaultAiQualityMode);
      dropdown.onChange(async (value) => {
        if (value === "efficient" || value === "full_vram_12gb") {
          settings.defaultAiQualityMode = value;
          await this.plugin.saveSettings();
        }
      });
    });
    this.renderMicDeviceControls(containerEl, settings);
    new import_obsidian.Setting(containerEl).setName("Auto summarize").setDesc("Call the local summarization endpoint after transcription.").addToggle(
      (toggle) => toggle.setValue(settings.autoSummarize).onChange(async (value) => {
        settings.autoSummarize = value;
        await this.plugin.saveSettings();
      })
    );
    new import_obsidian.Setting(containerEl).setName("Include full transcript in saved note").setDesc("If disabled, only the structured summary is saved.").addToggle(
      (toggle) => toggle.setValue(settings.includeFullTranscript).onChange(async (value) => {
        settings.includeFullTranscript = value;
        await this.plugin.saveSettings();
      })
    );
    new import_obsidian.Setting(containerEl).setName("Capture profile").setDesc("Fast = lower latency, Balanced = default, Pristine = highest quality.").addDropdown((dropdown) => {
      dropdown.addOption("fast", "Fast");
      dropdown.addOption("balanced", "Balanced");
      dropdown.addOption("pristine", "Pristine");
      dropdown.setValue(settings.captureProfile);
      dropdown.onChange(async (value) => {
        if (value === "fast" || value === "balanced" || value === "pristine") {
          settings.captureProfile = value;
        }
        await this.plugin.saveSettings();
      });
    });
    new import_obsidian.Setting(containerEl).setName("Enable diarization").setDesc("Use local speaker diarization with multiple backends and fallback clustering.").addToggle(
      (toggle) => toggle.setValue(settings.enableDiarization).onChange(async (value) => {
        settings.enableDiarization = value;
        await this.plugin.saveSettings();
      })
    );
    new import_obsidian.Setting(containerEl).setName("Expected speaker count").setDesc("Guides diarization clustering (1-8). Use 3 for most meetings with several participants.").addSlider((slider) => {
      slider.setLimits(1, 8, 1);
      slider.setValue(settings.expectedSpeakers);
      slider.setDynamicTooltip();
      slider.onChange(async (value) => {
        settings.expectedSpeakers = value;
        await this.plugin.saveSettings();
      });
    });
    new import_obsidian.Setting(containerEl).setName("Prefer loopback + mic capture").setDesc("On Windows, try WASAPI loopback + microphone mixing first.").addToggle(
      (toggle) => toggle.setValue(settings.preferLoopbackCapture).onChange(async (value) => {
        settings.preferLoopbackCapture = value;
        await this.plugin.saveSettings();
      })
    );
  }
  renderMicDeviceControls(containerEl, settings) {
    const section = containerEl.createDiv();
    section.createEl("p", { text: "Loading microphone devices..." });
    void (async () => {
      try {
        const devices = await this.plugin.fetchMicDevices();
        section.empty();
        new import_obsidian.Setting(section).setName("Default microphone device").setDesc("This microphone is preselected as default for new sessions.").addDropdown((dropdown) => {
          dropdown.addOption("", "Default microphone");
          for (const mic of devices) {
            dropdown.addOption(mic.id, mic.name);
          }
          dropdown.setValue(settings.defaultMicDeviceContains || "");
          dropdown.onChange(async (value) => {
            settings.defaultMicDeviceContains = value;
            await this.plugin.saveSettings();
          });
        });
        new import_obsidian.Setting(section).setName("Microphone test").setDesc("Runs a 2-second local signal check for the selected default mic.").addButton(
          (btn) => btn.setButtonText("Test microphone").onClick(async () => {
            btn.setDisabled(true);
            btn.setButtonText("Testing...");
            try {
              const result = await this.plugin.testMicDevice(settings.defaultMicDeviceContains || "");
              if (!result.ok) {
                new import_obsidian.Notice(`Mic test failed: ${result.error || "Unknown error"}`);
              } else {
                new import_obsidian.Notice(
                  `Mic OK (${result.device_name}) - RMS ${result.rms.toFixed(1)}, Peak ${result.peak.toFixed(1)}`
                );
              }
            } catch (error) {
              console.error(error);
              new import_obsidian.Notice("Mic test failed: Companion server not reachable.");
            } finally {
              btn.setDisabled(false);
              btn.setButtonText("Test microphone");
            }
          })
        );
        new import_obsidian.Setting(section).setName("Fallback mic filter (manual)").setDesc("Used if dropdown detection is incomplete; set partial device name.").addText(
          (text) => text.setPlaceholder("e.g. rode, shure, headset").setValue(settings.defaultMicDeviceContains).onChange(async (value) => {
            settings.defaultMicDeviceContains = value.trim();
            await this.plugin.saveSettings();
          })
        );
      } catch (error) {
        console.error(error);
        section.empty();
        section.createEl("p", { text: "Could not load microphone devices. Companion server may be offline." });
      }
    })();
  }
  renderLlmControls(containerEl, settings) {
    const section = containerEl.createDiv();
    section.createEl("p", { text: "Loading Ollama models..." });
    void (async () => {
      try {
        const models = await this.plugin.fetchLlmModels(settings.ollamaBaseUrl);
        section.empty();
        new import_obsidian.Setting(section).setName("Summary model (LLM)").setDesc("Model used for local LLM tasks (currently meeting summarization).").addDropdown((dropdown) => {
          dropdown.addOption("", "Structured fallback only (no LLM)");
          for (const model of models) {
            dropdown.addOption(model.id, model.name);
          }
          dropdown.setValue(settings.defaultSummaryModel || "");
          dropdown.onChange(async (value) => {
            settings.defaultSummaryModel = value;
            await this.plugin.saveSettings();
          });
        }).addButton(
          (btn) => btn.setButtonText("Reload models").onClick(async () => {
            this.display();
          })
        );
      } catch (error) {
        console.error(error);
        section.empty();
        section.createEl("p", {
          text: "Could not load Ollama models. Check Ollama URL and that Ollama is running."
        });
      }
    })();
  }
  renderLanguageControls(containerEl, settings) {
    const section = containerEl.createDiv();
    section.createEl("p", { text: "Loading supported transcription languages..." });
    void (async () => {
      try {
        const languages = await this.plugin.fetchSupportedLanguages();
        section.empty();
        new import_obsidian.Setting(section).setName("Default meeting language").setDesc("Used as prefilled value in the session dialog. Auto works best for Denglisch/code-switching.").addDropdown((dropdown) => {
          for (const language of languages) {
            dropdown.addOption(language.code, language.label);
          }
          const knownCodes = new Set(languages.map((item) => item.code));
          const selected = knownCodes.has(settings.defaultMeetingLanguage) ? settings.defaultMeetingLanguage : "auto";
          dropdown.setValue(selected);
          dropdown.onChange(async (value) => {
            settings.defaultMeetingLanguage = value || "auto";
            await this.plugin.saveSettings();
          });
        });
      } catch (error) {
        console.error(error);
        section.empty();
        section.createEl("p", { text: "Could not load language list. Falling back to auto detection." });
      }
    })();
  }
};
var SpeakerMapModal = class extends import_obsidian.Modal {
  constructor(app, speakers, samples, audioSamples, ownSpeakerName) {
    super(app);
    this.speakerMap = /* @__PURE__ */ new Map();
    this.submitted = false;
    this.speakers = speakers;
    this.samples = samples;
    this.audioSamples = audioSamples;
    this.ownSpeakerName = ownSpeakerName?.trim() || "Me";
  }
  openAndGetResult() {
    this.open();
    return new Promise((resolve) => {
      this.resolver = resolve;
    });
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.createEl("h2", { text: "Identify Speakers" });
    contentEl.createEl("p", { text: "Rename each detected speaker before saving." });
    if (this.speakers.length === 0) {
      contentEl.createEl("p", { text: "No speaker diarization labels were detected." });
    }
    this.speakers.forEach((speakerId) => {
      const defaultLabel = speakerId === "SELF_USER" ? this.ownSpeakerName : speakerId;
      this.speakerMap.set(speakerId, defaultLabel);
      const row = contentEl.createDiv({ cls: "meeting-scribe-speaker-row" });
      row.style.marginBottom = "14px";
      const sampleText = this.samples[speakerId] || "No sample available.";
      row.createEl("small", { text: `"${sampleText}"` }).style.fontStyle = "italic";
      const audioBase64 = this.audioSamples[speakerId];
      if (audioBase64) {
        const audioEl = row.createEl("audio");
        audioEl.controls = true;
        audioEl.preload = "none";
        audioEl.src = `data:audio/wav;base64,${audioBase64}`;
        audioEl.style.display = "block";
        audioEl.style.marginTop = "8px";
      }
      new import_obsidian.Setting(row).setName(speakerId).setDesc("Enter the real name").addText(
        (text) => text.setPlaceholder("Speaker name").setValue(defaultLabel).onChange((value) => this.speakerMap.set(speakerId, value.trim() || speakerId))
      );
    });
    new import_obsidian.Setting(contentEl).addButton(
      (btn) => btn.setButtonText("Cancel").onClick(() => {
        this.submitted = false;
        this.close();
      })
    ).addButton(
      (btn) => btn.setButtonText("Continue").setCta().onClick(() => {
        this.submitted = true;
        this.close();
      })
    );
  }
  onClose() {
    this.contentEl.empty();
    if (this.resolver) {
      this.resolver(this.submitted ? this.speakerMap : null);
    }
  }
};
var SaveFileModal = class extends import_obsidian.Modal {
  constructor(app, defaultFolder) {
    super(app);
    this.filename = defaultMeetingFileName();
    this.submitted = false;
    this.folder = defaultFolder || "/";
  }
  openAndGetResult() {
    this.open();
    return new Promise((resolve) => {
      this.resolver = resolve;
    });
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.createEl("h2", { text: "Save Meeting Notes" });
    new import_obsidian.Setting(contentEl).setName("Filename").addText((text) => text.setValue(this.filename).onChange((val) => this.filename = val));
    new import_obsidian.Setting(contentEl).setName("Folder Path").setDesc("Example: Meetings/Work (use / for vault root)").addText((text) => text.setValue(this.folder).onChange((val) => this.folder = val));
    const knownFolders = this.app.vault.getAllLoadedFiles().filter((file) => file instanceof import_obsidian.TFolder).map((folder) => folder.path).filter((path) => path.length > 0).sort();
    if (knownFolders.length > 0) {
      new import_obsidian.Setting(contentEl).setName("Quick pick existing folder").addDropdown((dropdown) => {
        dropdown.addOption("/", "/ (vault root)");
        for (const folderPath of knownFolders) {
          dropdown.addOption(folderPath, folderPath);
        }
        dropdown.setValue(this.folder);
        dropdown.onChange((value) => {
          this.folder = value;
        });
      });
    }
    new import_obsidian.Setting(contentEl).addButton(
      (btn) => btn.setButtonText("Cancel").onClick(() => {
        this.submitted = false;
        this.close();
      })
    ).addButton(
      (btn) => btn.setButtonText("Save").setCta().onClick(() => {
        this.submitted = true;
        this.close();
      })
    );
  }
  onClose() {
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
      folder: this.folder.trim() || "/"
    });
  }
};
var SessionConfigModal = class extends import_obsidian.Modal {
  constructor(app, settings, micDevices, languageOptions, llmModelOptions) {
    super(app);
    this.submitted = false;
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
      meetingLanguage: settings.defaultMeetingLanguage || "auto"
    };
  }
  openAndGetResult() {
    this.open();
    return new Promise((resolve) => {
      this.resolver = resolve;
    });
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.createEl("h2", { text: "Session Settings" });
    contentEl.createEl("p", { text: "Configure this capture session before recording starts." });
    new import_obsidian.Setting(contentEl).setName("Meeting language").setDesc("Select forced transcription language, or Auto for mixed-language speech (recommended for Denglisch).").addDropdown((dropdown) => {
      for (const language of this.languageOptions) {
        dropdown.addOption(language.code, language.label);
      }
      const knownCodes = new Set(this.languageOptions.map((item) => item.code));
      const selected = knownCodes.has(this.options.meetingLanguage) ? this.options.meetingLanguage : "auto";
      dropdown.setValue(selected);
      dropdown.onChange((value) => {
        this.options.meetingLanguage = value || "auto";
      });
    });
    new import_obsidian.Setting(contentEl).setName("Expected participants").setDesc("Helps diarization separate speakers better.").addSlider((slider) => {
      slider.setLimits(1, 8, 1);
      slider.setValue(this.options.expectedSpeakers);
      slider.setDynamicTooltip();
      slider.onChange((value) => {
        this.options.expectedSpeakers = value;
      });
    });
    new import_obsidian.Setting(contentEl).setName("Capture profile").setDesc("Fast = lower latency, Balanced = default, Pristine = highest quality.").addDropdown((dropdown) => {
      dropdown.addOption("fast", "Fast");
      dropdown.addOption("balanced", "Balanced");
      dropdown.addOption("pristine", "Pristine");
      dropdown.setValue(this.options.captureProfile);
      dropdown.onChange((value) => {
        if (value === "fast" || value === "balanced" || value === "pristine") {
          this.options.captureProfile = value;
        }
      });
    });
    new import_obsidian.Setting(contentEl).setName("AI quality mode").setDesc("Use full 12GB mode for stronger local ASR and speaker separation.").addDropdown((dropdown) => {
      dropdown.addOption("efficient", "Efficient (lower VRAM)");
      dropdown.addOption("full_vram_12gb", "Full VRAM 12GB (best quality)");
      dropdown.setValue(this.options.aiQualityMode);
      dropdown.onChange((value) => {
        if (value === "efficient" || value === "full_vram_12gb") {
          this.options.aiQualityMode = value;
        }
      });
    });
    new import_obsidian.Setting(contentEl).setName("Enable diarization").addToggle(
      (toggle) => toggle.setValue(this.options.enableDiarization).onChange((value) => {
        this.options.enableDiarization = value;
      })
    );
    new import_obsidian.Setting(contentEl).setName("Prefer loopback + mic capture").addToggle(
      (toggle) => toggle.setValue(this.options.preferLoopbackCapture).onChange((value) => {
        this.options.preferLoopbackCapture = value;
      })
    );
    new import_obsidian.Setting(contentEl).setName("Meeting apps only (system audio filter)").setDesc("Use only system audio when selected meeting apps are actively outputting sound.").addToggle(
      (toggle) => toggle.setValue(this.options.appAudioOnly).onChange((value) => {
        this.options.appAudioOnly = value;
      })
    );
    new import_obsidian.Setting(contentEl).setName("Meeting app executables").setDesc("Comma-separated, e.g. discord.exe, teams.exe, zoom.exe").addText(
      (text) => text.setValue(this.options.targetApps).onChange((value) => this.options.targetApps = value.trim())
    );
    new import_obsidian.Setting(contentEl).setName("Include own microphone input").setDesc("If enabled, your mic is included in the recording even with meeting-app filter enabled.").addToggle(
      (toggle) => toggle.setValue(this.options.includeMicInput).onChange((value) => {
        this.options.includeMicInput = value;
      })
    );
    new import_obsidian.Setting(contentEl).setName("Own speaker name").setDesc("Prelabel mic-dominant speech as this name.").addText(
      (text) => text.setValue(this.options.ownSpeakerName).onChange((value) => this.options.ownSpeakerName = value.trim() || "Me")
    );
    if (this.micDevices.length > 0) {
      new import_obsidian.Setting(contentEl).setName("Microphone device").setDesc("Select the microphone to use for your own speech.").addDropdown((dropdown) => {
        dropdown.addOption("", "Default microphone");
        for (const mic of this.micDevices) {
          dropdown.addOption(mic.id, mic.name);
        }
        dropdown.setValue(this.options.micDeviceContains || "");
        dropdown.onChange((value) => {
          this.options.micDeviceContains = value;
        });
      });
    } else {
      new import_obsidian.Setting(contentEl).setName("Microphone device filter").setDesc("Could not load mic list; use partial device name.").addText(
        (text) => text.setValue(this.options.micDeviceContains).onChange((value) => this.options.micDeviceContains = value.trim())
      );
    }
    new import_obsidian.Setting(contentEl).setName("Auto summarize").addToggle(
      (toggle) => toggle.setValue(this.options.autoSummarize).onChange((value) => {
        this.options.autoSummarize = value;
      })
    );
    new import_obsidian.Setting(contentEl).setName("Summary model for this session").setDesc("Overrides the default LLM model only for this capture session.").addDropdown((dropdown) => {
      dropdown.addOption("", "Use default summary model");
      for (const model of this.llmModelOptions) {
        dropdown.addOption(model.id, model.name);
      }
      if (this.options.summaryModel && !this.llmModelOptions.some((model) => model.id === this.options.summaryModel)) {
        dropdown.addOption(this.options.summaryModel, `${this.options.summaryModel} (custom)`);
      }
      dropdown.setValue(this.options.summaryModel || "");
      dropdown.onChange((value) => {
        this.options.summaryModel = value;
      });
    });
    new import_obsidian.Setting(contentEl).setName("Include full transcript in final note").addToggle(
      (toggle) => toggle.setValue(this.options.includeFullTranscript).onChange((value) => {
        this.options.includeFullTranscript = value;
      })
    );
    new import_obsidian.Setting(contentEl).addButton(
      (btn) => btn.setButtonText("Cancel").onClick(() => {
        this.submitted = false;
        this.close();
      })
    ).addButton(
      (btn) => btn.setButtonText("Start Capture").setCta().onClick(() => {
        this.submitted = true;
        this.close();
      })
    );
  }
  onClose() {
    this.contentEl.empty();
    if (!this.resolver) {
      return;
    }
    this.resolver(this.submitted ? this.options : null);
  }
};
