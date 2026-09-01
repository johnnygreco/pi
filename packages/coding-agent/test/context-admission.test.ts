import { mkdirSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import {
	type AssistantMessage,
	type AssistantMessageEvent,
	EventStream,
	getModel,
	type ToolResultMessage,
} from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AgentSession, type ContextAdmission, ContextAdmissionDeniedError } from "../src/core/agent-session.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import type { InlineExtension } from "../src/core/extensions/index.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
import { createTestExtensionsResult, createTestResourceLoader } from "./utilities.ts";

class MockAssistantStream extends EventStream<AssistantMessageEvent, AssistantMessage> {
	constructor() {
		super(
			(event) => event.type === "done" || event.type === "error",
			(event) => {
				if (event.type === "done") return event.message;
				if (event.type === "error") return event.error;
				throw new Error("Unexpected event type");
			},
		);
	}
}

function assistantMessage(
	content: AssistantMessage["content"],
	stopReason: Extract<AssistantMessage["stopReason"], "deferred" | "length" | "stop" | "toolUse"> = "stop",
) {
	return {
		role: "assistant" as const,
		content,
		api: "anthropic-messages" as const,
		provider: "anthropic",
		model: "mock",
		usage: {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		},
		stopReason,
		timestamp: Date.now(),
	} satisfies AssistantMessage;
}

function completedStream(message: AssistantMessage): MockAssistantStream {
	const stream = new MockAssistantStream();
	queueMicrotask(() => {
		stream.push({ type: "start", partial: { ...message, content: [] } });
		stream.push({ type: "done", reason: message.stopReason === "toolUse" ? "toolUse" : "stop", message });
	});
	return stream;
}

describe("managed context admission", () => {
	let tempDir: string;
	let session: AgentSession | undefined;

	beforeEach(() => {
		tempDir = join(tmpdir(), `pi-context-admission-${Date.now()}-${Math.random().toString(36).slice(2)}`);
		mkdirSync(tempDir, { recursive: true });
	});

	afterEach(() => {
		session?.dispose();
		rmSync(tempDir, { recursive: true, force: true });
	});

	async function createSession(
		streamFn: ConstructorParameters<typeof Agent>[0]["streamFn"],
		contextAdmission: ContextAdmission,
		baseToolsOverride?: Record<string, AgentTool>,
		extensionFactories?: InlineExtension[],
	) {
		const model = getModel("anthropic", "claude-sonnet-4-5")!;
		const agent = new Agent({
			initialState: { model, systemPrompt: "Test", tools: [] },
			streamFn,
		});
		const authStorage = AuthStorage.create(join(tempDir, "auth.json"));
		const modelRegistry = await createModelRegistry(authStorage, tempDir);
		await authStorage.modify("anthropic", async () => ({ type: "api_key", key: "test-key" }));
		const sessionManager = SessionManager.create(tempDir, join(tempDir, "sessions"));
		const extensionsResult = extensionFactories
			? await createTestExtensionsResult(extensionFactories, tempDir)
			: undefined;
		session = new AgentSession({
			agent,
			sessionManager,
			settingsManager: SettingsManager.create(tempDir, tempDir),
			cwd: tempDir,
			modelRuntime: getModelRuntime(modelRegistry),
			resourceLoader: createTestResourceLoader(extensionsResult ? { extensionsResult } : undefined),
			contextAdmission,
			baseToolsOverride,
			initialActiveToolNames: baseToolsOverride ? Object.keys(baseToolsOverride) : [],
		});
		return { session, sessionManager };
	}

	it("admits a rendered idle prompt before context and external persistence", async () => {
		const providerInputs: string[] = [];
		const { session, sessionManager } = await createSession(
			(_model, context) => {
				providerInputs.push(JSON.stringify(context.messages));
				return completedStream(assistantMessage([{ type: "text", text: "done" }]));
			},
			{
				admitUserMessage: async (message) => ({
					action: "allow",
					message: { ...message, content: [{ type: "text", text: "admitted prompt" }] },
				}),
				admitToolResult: async () => ({ action: "allow" }),
				admitProviderContext: async () => ({ action: "allow" }),
			},
		);

		await session.prompt("raw prompt");

		expect(providerInputs[0]).toContain("admitted prompt");
		expect(providerInputs[0]).not.toContain("raw prompt");
		expect(JSON.stringify(sessionManager.getEntries())).toContain("admitted prompt");
		expect(JSON.stringify(sessionManager.getEntries())).not.toContain("raw prompt");
	});

	it("keeps standard input extensions before mandatory admission", async () => {
		const admittedInputs: string[] = [];
		const providerInputs: string[] = [];
		const { session } = await createSession(
			(_model, context) => {
				providerInputs.push(JSON.stringify(context.messages));
				return completedStream(assistantMessage([{ type: "text", text: "done" }]));
			},
			{
				admitUserMessage: async (message) => {
					admittedInputs.push(JSON.stringify(message.content));
					return { action: "allow" };
				},
				admitToolResult: async () => ({ action: "allow" }),
				admitProviderContext: async () => ({ action: "allow" }),
			},
			undefined,
			[
				(pi) => {
					pi.on("input", (event) => ({ action: "transform", text: `${event.text} transformed` }));
				},
			],
		);

		await session.prompt("raw prompt");

		expect(admittedInputs).toHaveLength(1);
		expect(admittedInputs[0]).toContain("raw prompt transformed");
		expect(providerInputs[0]).toContain("raw prompt transformed");
	});

	it("does not append a denied idle prompt to context or JSONL", async () => {
		let providerCalls = 0;
		const { session, sessionManager } = await createSession(
			() => {
				providerCalls++;
				return completedStream(assistantMessage([{ type: "text", text: "done" }]));
			},
			{
				admitUserMessage: async (message) =>
					JSON.stringify(message).includes("denied")
						? { action: "deny", reason: "test policy" }
						: { action: "allow" },
				admitToolResult: async () => ({ action: "allow" }),
				admitProviderContext: async () => ({ action: "allow" }),
			},
		);

		await session.prompt("allowed");
		const sessionFile = sessionManager.getSessionFile();
		if (!sessionFile) throw new Error("Expected a persisted session file");
		const beforeDenial = readFileSync(sessionFile, "utf8");

		const denied = session.prompt("denied idle prompt");
		await expect(denied).rejects.toBeInstanceOf(ContextAdmissionDeniedError);
		await expect(denied).rejects.toThrow("test policy");

		expect(providerCalls).toBe(1);
		expect(JSON.stringify(session.messages)).not.toContain("denied idle prompt");
		expect(readFileSync(sessionFile, "utf8")).toBe(beforeDenial);
	});

	it("does not queue or persist denied steering and follow-up prompts", async () => {
		let releaseFirstResponse: (() => void) | undefined;
		let providerCalls = 0;
		const { session, sessionManager } = await createSession(
			() => {
				providerCalls++;
				const stream = new MockAssistantStream();
				releaseFirstResponse = () => {
					const message = assistantMessage([{ type: "text", text: "done" }]);
					stream.push({ type: "start", partial: { ...message, content: [] } });
					stream.push({ type: "done", reason: "stop", message });
				};
				return stream;
			},
			{
				admitUserMessage: async (message) =>
					JSON.stringify(message).includes("denied") ? { action: "deny" } : { action: "allow" },
				admitToolResult: async () => ({ action: "allow" }),
				admitProviderContext: async () => ({ action: "allow" }),
			},
		);

		const firstPrompt = session.prompt("allowed");
		while (!releaseFirstResponse) {
			await new Promise((resolve) => setTimeout(resolve, 0));
		}
		await expect(session.prompt("denied steer", { streamingBehavior: "steer" })).rejects.toBeInstanceOf(
			ContextAdmissionDeniedError,
		);
		await expect(session.prompt("denied follow-up", { streamingBehavior: "followUp" })).rejects.toBeInstanceOf(
			ContextAdmissionDeniedError,
		);
		expect(session.pendingMessageCount).toBe(0);
		releaseFirstResponse();
		await firstPrompt;

		expect(providerCalls).toBe(1);
		expect(JSON.stringify(sessionManager.getEntries())).not.toContain("denied");
	});

	it.each(["steer", "followUp"] as const)(
		"admits a queued %s prompt before context and persistence",
		async (streamingBehavior) => {
			let releaseFirstResponse: (() => void) | undefined;
			const providerInputs: string[] = [];
			const { session, sessionManager } = await createSession(
				(_model, context) => {
					providerInputs.push(JSON.stringify(context.messages));
					if (providerInputs.length > 1) {
						return completedStream(assistantMessage([{ type: "text", text: "done" }]));
					}
					const stream = new MockAssistantStream();
					releaseFirstResponse = () => {
						const message = assistantMessage([{ type: "text", text: "first done" }]);
						stream.push({ type: "start", partial: { ...message, content: [] } });
						stream.push({ type: "done", reason: "stop", message });
					};
					return stream;
				},
				{
					admitUserMessage: async (message) =>
						JSON.stringify(message).includes("raw queued")
							? {
									action: "allow",
									message: { ...message, content: [{ type: "text", text: "admitted queued" }] },
								}
							: { action: "allow" },
					admitToolResult: async () => ({ action: "allow" }),
					admitProviderContext: async () => ({ action: "allow" }),
				},
			);

			const firstPrompt = session.prompt("first");
			while (!releaseFirstResponse) {
				await new Promise((resolve) => setTimeout(resolve, 0));
			}
			await session.prompt("raw queued", { streamingBehavior });
			releaseFirstResponse();
			await firstPrompt;

			const visibleState = JSON.stringify({ providerInputs, entries: sessionManager.getEntries() });
			expect(providerInputs).toHaveLength(2);
			expect(visibleState).toContain("admitted queued");
			expect(visibleState).not.toContain("raw queued");
		},
	);

	it("replaces a denied tool result before events, context, and persistence", async () => {
		const rawToolResult = "private tool bytes";
		const providerInputs: string[] = [];
		const emittedToolResults: ToolResultMessage[] = [];
		const tool: AgentTool = {
			name: "probe",
			label: "probe",
			description: "probe",
			parameters: Type.Object({}),
			execute: async () => ({ content: [{ type: "text", text: rawToolResult }], details: { rawToolResult } }),
		};
		const { session, sessionManager } = await createSession(
			(_model, context) => {
				providerInputs.push(JSON.stringify(context.messages));
				return context.messages.some((message) => message.role === "toolResult")
					? completedStream(assistantMessage([{ type: "text", text: "done" }]))
					: completedStream(
							assistantMessage([{ type: "toolCall", id: "tool-1", name: "probe", arguments: {} }], "toolUse"),
						);
			},
			{
				admitUserMessage: async () => ({ action: "allow" }),
				admitToolResult: async () => ({ action: "deny" }),
				admitProviderContext: async () => ({ action: "allow" }),
			},
			{ probe: tool },
		);
		session.subscribe((event) => {
			if (event.type === "message_end" && event.message.role === "toolResult") {
				emittedToolResults.push(event.message);
			}
		});

		await session.prompt("run probe");

		const visibleState = JSON.stringify({
			providerInputs,
			emittedToolResults,
			entries: sessionManager.getEntries(),
		});
		expect(visibleState).not.toContain(rawToolResult);
		expect(emittedToolResults).toHaveLength(1);
		expect(emittedToolResults[0].isError).toBe(true);
		expect(visibleState).toContain("[Tool result blocked by context admission]");
	});

	it("fails closed when user admission throws", async () => {
		let providerCalls = 0;
		const { session, sessionManager } = await createSession(
			() => {
				providerCalls++;
				return completedStream(assistantMessage([{ type: "text", text: "done" }]));
			},
			{
				admitUserMessage: async () => {
					throw new Error("admission unavailable");
				},
				admitToolResult: async () => ({ action: "allow" }),
				admitProviderContext: async () => ({ action: "allow" }),
			},
		);

		await expect(session.prompt("raw prompt")).rejects.toThrow("admission unavailable");
		expect(providerCalls).toBe(0);
		expect(JSON.stringify(sessionManager.getEntries())).not.toContain("raw prompt");
	});
});
