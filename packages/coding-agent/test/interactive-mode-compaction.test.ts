import type { Usage } from "@earendil-works/pi-ai";
import { Container } from "@earendil-works/pi-tui";
import { describe, expect, test, vi } from "vitest";
import { ContextAdmissionDeniedError } from "../src/core/agent-session.ts";
import type { SessionEntry } from "../src/core/session-manager.ts";
import { InteractiveMode } from "../src/modes/interactive/interactive-mode.ts";
import { initTheme } from "../src/modes/interactive/theme/theme.ts";
import { stripAnsi } from "../src/utils/ansi.ts";

describe("InteractiveMode compaction events", () => {
	test("uses the cache miss notice setting for compaction and branch summary costs", () => {
		const usage: Usage = {
			input: 10,
			output: 20,
			cacheRead: 30,
			cacheWrite: 40,
			totalTokens: 100,
			cost: { input: 0.01, output: 0.02, cacheRead: 0.03, cacheWrite: 0.065, total: 0.125 },
		};
		const addCompactionCostNotice = Reflect.get(InteractiveMode.prototype, "addCompactionCostNotice") as (
			this: { chatContainer: Container; settingsManager: { getShowCacheMissNotices(): boolean } },
			notice: {
				type: "compaction_cost";
				kind: "compaction" | "branch_summary";
				usage: Usage;
			},
		) => void;

		initTheme("dark");
		const enabled = {
			chatContainer: new Container(),
			settingsManager: { getShowCacheMissNotices: () => true },
		};
		addCompactionCostNotice.call(enabled, { type: "compaction_cost", kind: "compaction", usage });
		addCompactionCostNotice.call(enabled, {
			type: "compaction_cost",
			kind: "branch_summary",
			usage,
		});
		const output = stripAnsi(enabled.chatContainer.render(120).join("\n"));
		expect(output).toContain("Compaction: 100 tokens billed (~$0.13)");
		expect(output).toContain("Branch summary: 100 tokens billed (~$0.13)");

		const disabled = {
			chatContainer: new Container(),
			settingsManager: { getShowCacheMissNotices: () => false },
		};
		addCompactionCostNotice.call(disabled, { type: "compaction_cost", kind: "compaction", usage });
		expect(disabled.chatContainer.children).toHaveLength(0);
	});

	test("renders each compaction cost after its summary", () => {
		const currentUsage: Usage = {
			input: 10,
			output: 20,
			cacheRead: 30,
			cacheWrite: 40,
			totalTokens: 100,
			cost: { input: 0.01, output: 0.02, cacheRead: 0.03, cacheWrite: 0.04, total: 0.1 },
		};
		const previousUsage: Usage = {
			input: 1,
			output: 2,
			cacheRead: 3,
			cacheWrite: 4,
			totalTokens: 10,
			cost: { input: 0.001, output: 0.002, cacheRead: 0.003, cacheWrite: 0.004, total: 0.01 },
		};
		const entries: SessionEntry[] = [
			{
				type: "compaction",
				id: "current",
				parentId: "previous",
				timestamp: "2025-01-02T00:00:00Z",
				summary: "current summary",
				firstKeptEntryId: "kept",
				tokensBefore: 200,
				usage: currentUsage,
			},
			{
				type: "compaction",
				id: "previous",
				parentId: null,
				timestamp: "2025-01-01T00:00:00Z",
				summary: "previous summary",
				firstKeptEntryId: "kept",
				tokensBefore: 100,
				usage: previousUsage,
			},
		];
		const fakeThis = { renderSessionItems: vi.fn() };
		const renderSessionEntries = Reflect.get(InteractiveMode.prototype, "renderSessionEntries") as (
			this: typeof fakeThis,
			entries: SessionEntry[],
		) => void;

		renderSessionEntries.call(fakeThis, entries);

		expect(fakeThis.renderSessionItems).toHaveBeenCalledWith(
			[
				expect.objectContaining({ role: "compactionSummary", summary: "current summary" }),
				{ type: "compaction_cost", kind: "compaction", usage: currentUsage },
				expect.objectContaining({ role: "compactionSummary", summary: "previous summary" }),
				{ type: "compaction_cost", kind: "compaction", usage: previousUsage },
			],
			{},
		);
	});

	test("renders retained entries and appends the latest summary cost at the bottom", async () => {
		const usage: Usage = {
			input: 10,
			output: 20,
			cacheRead: 30,
			cacheWrite: 40,
			totalTokens: 100,
			cost: { input: 0.01, output: 0.02, cacheRead: 0.03, cacheWrite: 0.065, total: 0.125 },
		};
		const latestCompaction: SessionEntry = {
			type: "compaction",
			id: "latest",
			parentId: "previous",
			timestamp: "2025-01-02T00:00:00Z",
			summary: "summary",
			firstKeptEntryId: "kept",
			tokensBefore: 123,
			usage,
		};
		const previousCompaction: SessionEntry = {
			type: "compaction",
			id: "previous",
			parentId: null,
			timestamp: "2025-01-01T00:00:00Z",
			summary: "previous summary",
			firstKeptEntryId: "kept",
			tokensBefore: 100,
			usage,
		};
		const fakeThis = {
			isInitialized: true,
			footer: { invalidate: vi.fn() },
			autoCompactionEscapeHandler: undefined as (() => void) | undefined,
			autoCompactionLoader: undefined,
			defaultEditor: {},
			statusContainer: { clear: vi.fn() },
			chatContainer: { clear: vi.fn() },
			sessionManager: { buildContextEntries: vi.fn().mockReturnValue([latestCompaction, previousCompaction]) },
			renderSessionEntries: vi.fn(),
			addMessageToChat: vi.fn(),
			addCompactionCostNotice: vi.fn(),
			showError: vi.fn(),
			showStatus: vi.fn(),
			clearStatusIndicator: vi.fn(),
			flushCompactionQueue: vi.fn().mockResolvedValue(undefined),
			settingsManager: { getShowTerminalProgress: () => false },
			ui: { requestRender: vi.fn(), terminal: { setProgress: vi.fn() } },
		};

		const handleEvent = Reflect.get(InteractiveMode.prototype, "handleEvent") as (
			this: typeof fakeThis,
			event: {
				type: "compaction_end";
				reason: "manual" | "threshold" | "overflow";
				result: { tokensBefore: number; summary: string; usage?: Usage } | undefined;
				aborted: boolean;
				willRetry: boolean;
				errorMessage?: string;
			},
		) => Promise<void>;

		await handleEvent.call(fakeThis, {
			type: "compaction_end",
			reason: "manual",
			result: {
				tokensBefore: 123,
				summary: "summary",
				usage,
			},
			aborted: false,
			willRetry: false,
		});

		expect(fakeThis.chatContainer.clear).toHaveBeenCalledTimes(1);
		expect(fakeThis.renderSessionEntries).toHaveBeenCalledWith([previousCompaction]);
		expect(fakeThis.addMessageToChat).toHaveBeenCalledTimes(1);
		expect(fakeThis.addMessageToChat).toHaveBeenCalledWith(
			expect.objectContaining({
				role: "compactionSummary",
				tokensBefore: 123,
				summary: "summary",
			}),
		);
		expect(fakeThis.addCompactionCostNotice).toHaveBeenCalledWith({
			type: "compaction_cost",
			kind: "compaction",
			usage,
		});
		expect(fakeThis.flushCompactionQueue).toHaveBeenCalledWith({ willRetry: false });
	});

	test("preserves steering behavior when flushing into an active agent run", async () => {
		const fakeThis = {
			compactionQueuedMessages: [{ text: "change direction", mode: "steer" as const }],
			editor: { addToHistory: vi.fn() },
			session: {
				clearQueue: vi.fn(),
				prompt: vi.fn().mockImplementation(async (_text, options) => {
					(options as { preflightResult?: (success: boolean) => void })?.preflightResult?.(true);
				}),
				steer: vi.fn().mockResolvedValue(undefined),
				followUp: vi.fn().mockResolvedValue(undefined),
			},
			isExtensionCommand: vi.fn().mockReturnValue(false),
			updatePendingMessagesDisplay: vi.fn(),
			showError: vi.fn(),
		};

		const flushCompactionQueue = Reflect.get(InteractiveMode.prototype, "flushCompactionQueue") as (
			this: typeof fakeThis,
			options?: { willRetry?: boolean },
		) => Promise<void>;

		await flushCompactionQueue.call(fakeThis, { willRetry: false });

		expect(fakeThis.session.prompt).toHaveBeenCalledWith(
			"change direction",
			expect.objectContaining({ streamingBehavior: "steer" }),
		);
		expect(fakeThis.compactionQueuedMessages).toEqual([]);
		expect(fakeThis.showError).not.toHaveBeenCalled();
		expect(fakeThis.editor.addToHistory).toHaveBeenCalledWith("change direction");
	});

	test("drops a denied compaction message while preserving unrelated queued input", async () => {
		const denial = new ContextAdmissionDeniedError("blocked after compaction");
		const fakeThis = {
			compactionQueuedMessages: [
				{ text: "blocked message", mode: "steer" as const },
				{ text: "unrelated message", mode: "followUp" as const },
			],
			editor: { addToHistory: vi.fn() },
			session: {
				clearQueue: vi.fn(),
				prompt: vi.fn().mockImplementation(async (_text, options) => {
					(options as { preflightResult?: (success: boolean) => void })?.preflightResult?.(false);
					throw denial;
				}),
				steer: vi.fn().mockResolvedValue(undefined),
				followUp: vi.fn().mockResolvedValue(undefined),
			},
			isExtensionCommand: vi.fn().mockReturnValue(false),
			updatePendingMessagesDisplay: vi.fn(),
			showSubmissionError: vi.fn(),
		};
		const flushCompactionQueue = Reflect.get(InteractiveMode.prototype, "flushCompactionQueue") as (
			this: typeof fakeThis,
			options?: { willRetry?: boolean },
		) => Promise<void>;

		await expect(flushCompactionQueue.call(fakeThis, { willRetry: false })).resolves.toBeUndefined();

		expect(fakeThis.compactionQueuedMessages).toEqual([{ text: "unrelated message", mode: "followUp" }]);
		expect(fakeThis.session.clearQueue).not.toHaveBeenCalled();
		expect(fakeThis.editor.addToHistory).not.toHaveBeenCalledWith("blocked message");
		expect(fakeThis.editor.addToHistory).not.toHaveBeenCalledWith("unrelated message");
		expect(fakeThis.showSubmissionError).toHaveBeenCalledWith(denial, "Failed to send queued messages");
	});

	test("does not restore accepted messages when a later compaction message is denied", async () => {
		const denial = new ContextAdmissionDeniedError("blocked later message");
		const fakeThis = {
			compactionQueuedMessages: [
				{ text: "accepted first", mode: "steer" as const },
				{ text: "blocked second", mode: "followUp" as const },
				{ text: "pending third", mode: "steer" as const },
			],
			editor: { addToHistory: vi.fn() },
			session: {
				clearQueue: vi.fn(),
				prompt: vi.fn().mockImplementation(async (_text, options) => {
					(options as { preflightResult?: (success: boolean) => void })?.preflightResult?.(true);
				}),
				steer: vi.fn().mockResolvedValue(undefined),
				followUp: vi.fn().mockRejectedValue(denial),
			},
			isExtensionCommand: vi.fn().mockReturnValue(false),
			updatePendingMessagesDisplay: vi.fn(),
			showSubmissionError: vi.fn(),
		};
		const flushCompactionQueue = Reflect.get(InteractiveMode.prototype, "flushCompactionQueue") as (
			this: typeof fakeThis,
			options?: { willRetry?: boolean },
		) => Promise<void>;

		await expect(flushCompactionQueue.call(fakeThis, { willRetry: false })).resolves.toBeUndefined();

		expect(fakeThis.compactionQueuedMessages).toEqual([{ text: "pending third", mode: "steer" }]);
		expect(fakeThis.session.clearQueue).not.toHaveBeenCalled();
		expect(fakeThis.editor.addToHistory).toHaveBeenCalledWith("accepted first");
		expect(fakeThis.editor.addToHistory).not.toHaveBeenCalledWith("blocked second");
		expect(fakeThis.editor.addToHistory).not.toHaveBeenCalledWith("pending third");
		expect(fakeThis.session.steer).not.toHaveBeenCalled();
	});

	test("retries only the failed and unattempted suffix after an ordinary error", async () => {
		const failure = new Error("temporary queue failure");
		const fakeThis = {
			compactionQueuedMessages: [
				{ text: "accepted first", mode: "steer" as const },
				{ text: "failed second", mode: "followUp" as const },
				{ text: "pending third", mode: "steer" as const },
			],
			editor: { addToHistory: vi.fn() },
			session: {
				clearQueue: vi.fn(),
				prompt: vi.fn().mockImplementation(async (_text, options) => {
					(options as { preflightResult?: (success: boolean) => void })?.preflightResult?.(true);
				}),
				steer: vi.fn().mockResolvedValue(undefined),
				followUp: vi.fn().mockRejectedValue(failure),
			},
			isExtensionCommand: vi.fn().mockReturnValue(false),
			updatePendingMessagesDisplay: vi.fn(),
			showSubmissionError: vi.fn(),
		};
		const flushCompactionQueue = Reflect.get(InteractiveMode.prototype, "flushCompactionQueue") as (
			this: typeof fakeThis,
			options?: { willRetry?: boolean },
		) => Promise<void>;

		await expect(flushCompactionQueue.call(fakeThis, { willRetry: false })).resolves.toBeUndefined();

		expect(fakeThis.compactionQueuedMessages).toEqual([
			{ text: "failed second", mode: "followUp" },
			{ text: "pending third", mode: "steer" },
		]);
		expect(fakeThis.session.clearQueue).not.toHaveBeenCalled();
		expect(fakeThis.editor.addToHistory).toHaveBeenCalledWith("accepted first");
		expect(fakeThis.editor.addToHistory).not.toHaveBeenCalledWith("failed second");
		expect(fakeThis.editor.addToHistory).not.toHaveBeenCalledWith("pending third");
	});

	test("does not mutate the queue when the accepted first prompt later fails", async () => {
		let rejectFirstPrompt: ((error: Error) => void) | undefined;
		const firstPromptFailure = new Promise<void>((_resolve, reject) => {
			rejectFirstPrompt = reject;
		});
		const fakeThis = {
			compactionQueuedMessages: [
				{ text: "accepted first", mode: "steer" as const },
				{ text: "accepted second", mode: "followUp" as const },
			],
			editor: { addToHistory: vi.fn() },
			session: {
				clearQueue: vi.fn(),
				prompt: vi.fn().mockImplementation(async (_text, options) => {
					(options as { preflightResult?: (success: boolean) => void })?.preflightResult?.(true);
					await firstPromptFailure;
				}),
				steer: vi.fn().mockResolvedValue(undefined),
				followUp: vi.fn().mockResolvedValue(undefined),
			},
			isExtensionCommand: vi.fn().mockReturnValue(false),
			updatePendingMessagesDisplay: vi.fn(),
			showSubmissionError: vi.fn(),
		};
		const flushCompactionQueue = Reflect.get(InteractiveMode.prototype, "flushCompactionQueue") as (
			this: typeof fakeThis,
			options?: { willRetry?: boolean },
		) => Promise<void>;

		await flushCompactionQueue.call(fakeThis, { willRetry: false });
		rejectFirstPrompt?.(new Error("provider failed after preflight"));
		await firstPromptFailure.catch(() => {});
		await Promise.resolve();

		expect(fakeThis.compactionQueuedMessages).toEqual([]);
		expect(fakeThis.session.clearQueue).not.toHaveBeenCalled();
		expect(fakeThis.editor.addToHistory).toHaveBeenCalledWith("accepted first");
		expect(fakeThis.editor.addToHistory).toHaveBeenCalledWith("accepted second");
		expect(fakeThis.showSubmissionError).toHaveBeenCalledWith(
			expect.objectContaining({ message: "provider failed after preflight" }),
			"Failed to send queued messages",
		);
	});
});
