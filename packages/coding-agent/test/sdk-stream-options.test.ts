import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type Api,
	type AssistantMessage,
	type Context,
	createAssistantMessageEventStream,
	type Model,
	type SimpleStreamOptions,
	type UserMessage,
} from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { ContextAdmission } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { type Settings, SettingsManager } from "../src/core/settings-manager.ts";

import { createModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";

describe("createAgentSession stream options", () => {
	let tempDir: string;
	let cwd: string;
	let agentDir: string;

	beforeEach(() => {
		tempDir = mkdtempSync(join(tmpdir(), "pi-sdk-stream-options-"));
		cwd = join(tempDir, "project");
		agentDir = join(tempDir, "agent");
		mkdirSync(cwd, { recursive: true });
		mkdirSync(agentDir, { recursive: true });
	});

	afterEach(() => {
		if (tempDir) {
			rmSync(tempDir, { recursive: true, force: true });
		}
	});

	function createModel(api: Api): Model<Api> {
		return {
			id: "capture-model",
			name: "Capture Model",
			api,
			provider: "capture-provider",
			baseUrl: "https://capture.invalid/v1",
			reasoning: false,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 128000,
			maxTokens: 4096,
			headers: { "x-model": "model" },
		};
	}

	function createDoneStream(api: Api) {
		const stream = createAssistantMessageEventStream();
		const message: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "ok" }],
			api,
			provider: "capture-provider",
			model: "capture-model",
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
			stopReason: "stop",
			timestamp: Date.now(),
		};
		stream.end(message);
		return stream;
	}

	async function captureStreamOptions(
		api: Api,
		settings: Partial<Settings>,
		requestOptions: SimpleStreamOptions = {},
		extensionSource?: string,
		contextAdmission?: ContextAdmission,
		context: Context = { messages: [] },
		captureContext?: (context: Context) => void,
	): Promise<SimpleStreamOptions | undefined> {
		const model = createModel(api);
		const settingsManager = SettingsManager.inMemory(settings);
		if (extensionSource) {
			const extensionsDir = join(agentDir, "extensions");
			mkdirSync(extensionsDir, { recursive: true });
			writeFileSync(join(extensionsDir, "headers.ts"), extensionSource);
		}

		const authStorage = AuthStorage.create(join(agentDir, "auth.json"));
		await authStorage.modify(model.provider, async () => ({ type: "api_key", key: "test-api-key" }));
		const modelRegistry = await createModelRegistry(authStorage, join(agentDir, "models.json"));
		let capturedOptions: SimpleStreamOptions | undefined;

		modelRegistry.registerProvider(model.provider, {
			api,
			headers: { "x-provider": "provider" },
			streamSimple: (_model, providerContext, providerOptions) => {
				captureContext?.(providerContext);
				capturedOptions = providerOptions;
				return createDoneStream(api);
			},
		});

		const modelRuntime = getModelRuntime(modelRegistry);
		const sessionManager = SessionManager.inMemory(cwd);
		const { session } = await createAgentSession({
			cwd,
			agentDir,
			model,
			modelRuntime,
			settingsManager,
			sessionManager,
			contextAdmission,
		});

		try {
			const stream = await session.agent.streamFunction(model, context, requestOptions);
			await stream.result();
			return capturedOptions;
		} finally {
			session.dispose();
			modelRegistry.unregisterProvider(model.provider);
		}
	}

	it("forwards httpIdleTimeoutMs as timeoutMs for OpenAI Codex", async () => {
		const options = await captureStreamOptions("openai-codex-responses", { httpIdleTimeoutMs: 1234 });

		expect(options?.timeoutMs).toBe(1234);
	});

	it("defaults timeoutMs from httpIdleTimeoutMs for all providers", async () => {
		const options = await captureStreamOptions("openai-completions", { httpIdleTimeoutMs: 1234 });

		expect(options?.timeoutMs).toBe(1234);
	});

	it("lets request timeoutMs override httpIdleTimeoutMs for OpenAI Codex", async () => {
		const options = await captureStreamOptions(
			"openai-codex-responses",
			{ httpIdleTimeoutMs: 1234 },
			{ timeoutMs: 0 },
		);

		expect(options?.timeoutMs).toBe(0);
	});

	it("forwards websocketConnectTimeoutMs from settings", async () => {
		const options = await captureStreamOptions("openai-codex-responses", { websocketConnectTimeoutMs: 1234 });

		expect(options?.websocketConnectTimeoutMs).toBe(1234);
	});

	it("lets request websocketConnectTimeoutMs override settings", async () => {
		const options = await captureStreamOptions(
			"openai-codex-responses",
			{ websocketConnectTimeoutMs: 1234 },
			{ websocketConnectTimeoutMs: 0 },
		);

		expect(options?.websocketConnectTimeoutMs).toBe(0);
	});

	it("forwards provider retry settings", async () => {
		const options = await captureStreamOptions("openai-completions", {
			retry: { provider: { maxRetries: 2, maxRetryDelayMs: 3000 } },
		});

		expect(options?.maxRetries).toBe(2);
		expect(options?.maxRetryDelayMs).toBe(3000);
	});

	it("runs before_provider_headers on assembled headers without forwarding the transform", async () => {
		const options = await captureStreamOptions(
			"openai-completions",
			{},
			{ headers: { "x-explicit": "explicit" } },
			`export default function (pi) {
				pi.on("before_provider_headers", (event) => {
					event.headers["x-hook"] = [
						event.headers["x-provider"],
						event.headers["x-model"],
						event.headers["x-explicit"],
					].join(":");
				});
			}`,
		);

		expect(options?.headers).toMatchObject({
			"x-provider": "provider",
			"x-model": "model",
			"x-explicit": "explicit",
			"x-hook": "provider:model:explicit",
		});
		expect(options).not.toHaveProperty("transformHeaders");
	});

	it("uses the outbound context to keep a queued admission handle off an earlier request retry", async () => {
		const firstMessage: UserMessage = {
			role: "user",
			content: [{ type: "text", text: "first" }],
			timestamp: 1,
		};
		const queuedMessage: UserMessage = {
			role: "user",
			content: [{ type: "text", text: "queued" }],
			timestamp: 2,
		};
		const handles = new Map<string, string>();
		let nextHandle = "handle-first";
		const admission: ContextAdmission = {
			admitUserMessage: async (message) => {
				handles.set(JSON.stringify(message.content), nextHandle);
				return { action: "allow" };
			},
			admitToolResult: async () => ({ action: "allow" }),
			admitProviderContext: async () => ({ action: "allow" }),
			transformProviderHeaders: async (headers, context) => {
				let candidate: (typeof context.messages)[number] | undefined;
				for (let i = context.messages.length - 1; i >= 0; i--) {
					const message = context.messages[i];
					if (message.role === "user" || message.role === "toolResult") {
						candidate = message;
						break;
					}
				}
				const handle = candidate ? handles.get(JSON.stringify(candidate.content)) : undefined;
				return handle ? { ...headers, "x-admission": handle } : headers;
			},
		};

		await admission.admitUserMessage(firstMessage, { source: "interactive" });
		nextHandle = "handle-queued";
		await admission.admitUserMessage(queuedMessage, { source: "interactive" });

		const options = await captureStreamOptions("openai-completions", {}, {}, undefined, admission, {
			messages: [firstMessage],
		});

		expect(options?.headers?.["x-admission"]).toBe("handle-first");
	});

	it("admits a generated provider context before serialization and header transformation", async () => {
		const generated: Context = {
			messages: [{ role: "user", content: [{ type: "text", text: "generated" }], timestamp: 1 }],
		};
		const admitted: Context = {
			messages: [{ role: "user", content: [{ type: "text", text: "admitted" }], timestamp: 1 }],
		};
		let serializedContext: Context | undefined;
		const admission: ContextAdmission = {
			admitUserMessage: async () => ({ action: "allow" }),
			admitToolResult: async () => ({ action: "allow" }),
			admitProviderContext: async (context) => {
				expect(context).toBe(generated);
				return { action: "allow", context: admitted };
			},
			transformProviderHeaders: async (headers, context) => {
				expect(context).toBe(admitted);
				return { ...headers, "x-admission": "generated-handle" };
			},
		};

		const options = await captureStreamOptions(
			"openai-completions",
			{},
			{},
			undefined,
			admission,
			generated,
			(context) => {
				serializedContext = context;
			},
		);

		expect(serializedContext).toBe(admitted);
		expect(options?.headers?.["x-admission"]).toBe("generated-handle");
	});

	it("does not call the provider when outbound context admission denies", async () => {
		let providerCalled = false;
		const admission: ContextAdmission = {
			admitUserMessage: async () => ({ action: "allow" }),
			admitToolResult: async () => ({ action: "allow" }),
			admitProviderContext: async () => ({ action: "deny", reason: "generated context denied" }),
		};

		await expect(
			captureStreamOptions("openai-completions", {}, {}, undefined, admission, { messages: [] }, () => {
				providerCalled = true;
			}),
		).rejects.toThrow("generated context denied");
		expect(providerCalled).toBe(false);
	});
});
