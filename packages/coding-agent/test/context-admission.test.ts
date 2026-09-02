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
				admitMessage: async (message) => ({
					action: "allow",
					message: { ...message, content: [{ type: "text", text: "admitted prompt" }] },
				}),
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
		const admittedOrigins: string[] = [];
		const providerInputs: string[] = [];
		const { session } = await createSession(
			(_model, context) => {
				providerInputs.push(JSON.stringify(context.messages));
				return completedStream(assistantMessage([{ type: "text", text: "done" }]));
			},
			{
				admitMessage: async (message, { origin }) => {
					admittedOrigins.push(origin);
					if (origin === "user") admittedInputs.push(JSON.stringify(message));
					return { action: "allow" };
				},
				admitProviderContext: async () => ({ action: "allow" }),
			},
			undefined,
			[
				(pi) => {
					pi.on("input", (event) => ({ action: "transform", text: `${event.text} transformed` }));
					pi.on("before_agent_start", () => ({
						message: { customType: "context", content: "extension context", display: false },
					}));
				},
			],
		);

		await session.prompt("raw prompt");

		expect(admittedInputs).toHaveLength(1);
		expect(admittedInputs[0]).toContain("raw prompt transformed");
		expect(admittedOrigins).toContain("extension_message");
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
				admitMessage: async (message) =>
					JSON.stringify(message).includes("denied")
						? { action: "deny", reason: "test policy" }
						: { action: "allow" },
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
				admitMessage: async (message) =>
					JSON.stringify(message).includes("denied") ? { action: "deny" } : { action: "allow" },
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
					admitMessage: async (message) =>
						JSON.stringify(message).includes("raw queued")
							? {
									action: "allow",
									message: { ...message, content: [{ type: "text", text: "admitted queued" }] },
								}
							: { action: "allow" },
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
				admitMessage: async (_message, { origin }) =>
					origin === "tool_result" ? { action: "deny" } : { action: "allow" },
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

	it("replaces denied assistant text while preserving and executing tool calls", async () => {
		const rawAssistantText = "poisoned assistant text";
		const providerInputs: string[] = [];
		let toolExecutions = 0;
		const tool: AgentTool = {
			name: "probe",
			label: "probe",
			description: "probe",
			parameters: Type.Object({}),
			execute: async () => {
				toolExecutions++;
				return { content: [{ type: "text", text: "safe result" }], details: {} };
			},
		};
		const { session, sessionManager } = await createSession(
			(_model, context) => {
				providerInputs.push(JSON.stringify(context.messages));
				return context.messages.some((message) => message.role === "toolResult")
					? completedStream(assistantMessage([{ type: "text", text: "done" }]))
					: completedStream(
							assistantMessage(
								[
									{ type: "thinking", thinking: "preserved reasoning" },
									{ type: "text", text: rawAssistantText },
									{ type: "toolCall", id: "tool-1", name: "probe", arguments: {} },
								],
								"toolUse",
							),
						);
			},
			{
				admitMessage: async (message, { origin }) =>
					origin === "assistant" && JSON.stringify(message).includes(rawAssistantText)
						? { action: "deny" }
						: { action: "allow" },
				admitProviderContext: async () => ({ action: "allow" }),
			},
			{ probe: tool },
		);

		await session.prompt("run probe");

		const visibleState = JSON.stringify({
			providerInputs,
			messages: session.messages,
			entries: sessionManager.getEntries(),
		});
		expect(toolExecutions).toBe(1);
		expect(visibleState).not.toContain(rawAssistantText);
		expect(visibleState).toContain("[Assistant message blocked by context admission]");
		expect(visibleState).toContain("preserved reasoning");
		expect(visibleState).toContain('"name":"probe"');
	});

	it("admits extension messages before every delivery mode queues or appends them", async () => {
		let releaseFirstResponse: (() => void) | undefined;
		let providerCalls = 0;
		const { session, sessionManager } = await createSession(
			() => {
				providerCalls++;
				if (providerCalls > 1) {
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
				admitMessage: async (message, { origin }) =>
					origin === "extension_message" && message.role === "custom"
						? { action: "allow", message: { ...message, content: "admitted extension" } }
						: { action: "allow" },
				admitProviderContext: async () => ({ action: "allow" }),
			},
		);

		const firstPrompt = session.prompt("start");
		while (!releaseFirstResponse) await new Promise((resolve) => setTimeout(resolve, 0));
		await session.sendCustomMessage({ customType: "a", content: "raw steer", display: false });
		await session.sendCustomMessage(
			{ customType: "b", content: "raw follow-up", display: false },
			{ deliverAs: "followUp" },
		);
		await session.sendCustomMessage(
			{ customType: "c", content: "raw deferred", display: false },
			{ triggerTurn: false },
		);
		await session.sendCustomMessage(
			{ customType: "d", content: "raw next turn", display: false },
			{ deliverAs: "nextTurn" },
		);
		releaseFirstResponse();
		await firstPrompt;
		await session.prompt("flush next turn");

		const visibleState = JSON.stringify({ messages: session.messages, entries: sessionManager.getEntries() });
		expect(visibleState).not.toContain("raw steer");
		expect(visibleState).not.toContain("raw follow-up");
		expect(visibleState).not.toContain("raw deferred");
		expect(visibleState).not.toContain("raw next turn");
		expect(visibleState.match(/admitted extension/g)).toHaveLength(8);
	});

	it("admits immediate and deferred bash results before context and persistence", async () => {
		let releaseResponse: (() => void) | undefined;
		const { session, sessionManager } = await createSession(
			() => {
				const stream = new MockAssistantStream();
				releaseResponse = () => {
					const message = assistantMessage([{ type: "text", text: "done" }]);
					stream.push({ type: "done", reason: "stop", message });
				};
				return stream;
			},
			{
				admitMessage: async (message, { origin }) =>
					origin === "bash_execution" && message.role === "bashExecution"
						? { action: "allow", message: { ...message, output: "admitted bash output" } }
						: { action: "allow" },
				admitProviderContext: async () => ({ action: "allow" }),
			},
		);

		await session.recordBashResult("idle", {
			output: "raw idle output",
			exitCode: 0,
			cancelled: false,
			truncated: false,
		});
		const prompt = session.prompt("start");
		while (!releaseResponse) await new Promise((resolve) => setTimeout(resolve, 0));
		await session.recordBashResult("deferred", {
			output: "raw deferred output",
			exitCode: 0,
			cancelled: false,
			truncated: false,
		});
		expect(session.hasPendingBashMessages).toBe(true);
		releaseResponse();
		await prompt;

		const visibleState = JSON.stringify({ messages: session.messages, entries: sessionManager.getEntries() });
		expect(visibleState).not.toContain("raw idle output");
		expect(visibleState).not.toContain("raw deferred output");
		expect(visibleState.match(/admitted bash output/g)).toHaveLength(4);
	});

	it("aborts compaction without appending a denied summary", async () => {
		const { session, sessionManager } = await createSession(
			() => completedStream(assistantMessage([{ type: "text", text: "unused" }])),
			{
				admitMessage: async (_message, { origin }) =>
					origin === "compaction_summary" ? { action: "deny", reason: "summary denied" } : { action: "allow" },
				admitProviderContext: async () => ({ action: "allow" }),
			},
			undefined,
			[
				(pi) => {
					pi.on("session_before_compact", (event) => ({
						compaction: {
							summary: "raw compaction summary",
							firstKeptEntryId: event.preparation.firstKeptEntryId,
							tokensBefore: event.preparation.tokensBefore,
						},
					}));
				},
			],
		);
		session.settingsManager.applyOverrides({ compaction: { keepRecentTokens: 1 } });
		const now = Date.now();
		sessionManager.appendMessage({
			role: "user",
			content: [{ type: "text", text: "message to compact" }],
			timestamp: now - 1000,
		});
		sessionManager.appendMessage(assistantMessage([{ type: "text", text: "assistant response to compact" }]));
		session.agent.state.messages = sessionManager.buildSessionContext().messages;
		const sessionFile = sessionManager.getSessionFile();
		if (!sessionFile) throw new Error("Expected a persisted session file");
		const beforeCompaction = readFileSync(sessionFile, "utf8");

		await expect(session.compact()).rejects.toThrow("summary denied");

		expect(readFileSync(sessionFile, "utf8")).toBe(beforeCompaction);
		expect(sessionManager.getEntries().some((entry) => entry.type === "compaction")).toBe(false);
	});

	it("persists an admitted replacement for an extension-provided branch summary", async () => {
		const { session, sessionManager } = await createSession(
			() => completedStream(assistantMessage([{ type: "text", text: "unused" }])),
			{
				admitMessage: async (message, { origin }) =>
					origin === "branch_summary" && message.role === "branchSummary"
						? { action: "allow", message: { ...message, summary: "admitted branch summary" } }
						: { action: "allow" },
				admitProviderContext: async () => ({ action: "allow" }),
			},
			undefined,
			[
				(pi) => {
					pi.on("session_before_tree", () => ({ summary: { summary: "raw branch summary" } }));
				},
			],
		);
		const targetId = sessionManager.appendMessage({
			role: "user",
			content: [{ type: "text", text: "target" }],
			timestamp: Date.now() - 2000,
		});
		sessionManager.appendMessage(assistantMessage([{ type: "text", text: "target response" }]));
		sessionManager.appendMessage({
			role: "user",
			content: [{ type: "text", text: "abandoned" }],
			timestamp: Date.now() - 1000,
		});
		sessionManager.appendMessage(assistantMessage([{ type: "text", text: "abandoned response" }]));
		session.agent.state.messages = sessionManager.buildSessionContext().messages;

		await session.navigateTree(targetId, { summarize: true });

		const visibleState = JSON.stringify({ messages: session.messages, entries: sessionManager.getEntries() });
		expect(visibleState).toContain("admitted branch summary");
		expect(visibleState).not.toContain("raw branch summary");
	});

	it("fails closed when user admission throws", async () => {
		let providerCalls = 0;
		const { session, sessionManager } = await createSession(
			() => {
				providerCalls++;
				return completedStream(assistantMessage([{ type: "text", text: "done" }]));
			},
			{
				admitMessage: async (_message, { origin }) => {
					if (origin !== "user") return { action: "allow" };
					throw new Error("admission unavailable");
				},
				admitProviderContext: async () => ({ action: "allow" }),
			},
		);

		await expect(session.prompt("raw prompt")).rejects.toThrow("admission unavailable");
		expect(providerCalls).toBe(0);
		expect(JSON.stringify(sessionManager.getEntries())).not.toContain("raw prompt");
	});
});
