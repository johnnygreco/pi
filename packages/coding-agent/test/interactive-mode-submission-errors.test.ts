import { describe, expect, it, vi } from "vitest";
import { ContextAdmissionDeniedError } from "../src/core/agent-session.ts";
import { InteractiveMode } from "../src/modes/interactive/interactive-mode.ts";

type SubmissionContext = {
	editor: { setText: (text: string) => void };
	showError: (message: string) => void;
	showSubmissionError: (error: unknown, prefix?: string) => void;
};

type EditorSubmitContext = SubmissionContext & {
	defaultEditor: { onSubmit?: (text: string) => void | Promise<void> };
	editor: SubmissionContext["editor"] & { addToHistory?: (text: string) => void };
	session: {
		isCompacting: boolean;
		isStreaming: boolean;
		isBashRunning: boolean;
		prompt: (text: string, options?: unknown) => Promise<void>;
	};
	flushPendingBashComponents: () => void;
	handleSubmission: (submit: () => Promise<void>, restoreText?: string) => Promise<void>;
	onInputCallback?: (text: string) => void;
	pendingUserInputs: string[];
};

type FollowUpContext = SubmissionContext & {
	editor: SubmissionContext["editor"] & {
		getText: () => string;
		getExpandedText?: () => string;
		addToHistory?: (text: string) => void;
	};
	session: {
		isCompacting: boolean;
		isStreaming: boolean;
		prompt: (text: string, options?: unknown) => Promise<void>;
	};
	queueCompactionMessage: (text: string, mode: "followUp") => void;
	isExtensionCommand: (text: string) => boolean;
	updatePendingMessagesDisplay: () => void;
	ui: { requestRender: () => void };
};

type InteractiveModePrivate = {
	showSubmissionError(this: SubmissionContext, error: unknown, prefix?: string): void;
	handleSubmission(this: SubmissionContext, submit: () => Promise<void>, restoreText?: string): Promise<void>;
	setupEditorSubmitHandler(this: EditorSubmitContext): void;
	handleFollowUp(this: FollowUpContext): Promise<void>;
};

const prototype = InteractiveMode.prototype as unknown as InteractiveModePrivate;

function submissionContext(): SubmissionContext {
	return {
		editor: { setText: vi.fn() },
		showError: vi.fn(),
		showSubmissionError: prototype.showSubmissionError,
	};
}

describe("InteractiveMode submission errors", () => {
	it("discards denied text but restores ordinary failed submissions", async () => {
		const context = submissionContext();

		await prototype.handleSubmission.call(
			context,
			async () => {
				throw new ContextAdmissionDeniedError("blocked by test policy");
			},
			"denied literal",
		);
		await prototype.handleSubmission.call(
			context,
			async () => {
				throw new Error("provider unavailable");
			},
			"retry me",
		);

		expect(context.editor.setText).toHaveBeenCalledWith("retry me");
		expect(context.editor.setText).not.toHaveBeenCalledWith("denied literal");
		expect(context.showError).toHaveBeenNthCalledWith(
			1,
			"Message denied by context admission: blocked by test policy",
		);
		expect(context.showError).toHaveBeenNthCalledWith(2, "provider unavailable");
	});

	it("consumes a rejected streaming editor submission and remains usable", async () => {
		const prompt = vi.fn<(text: string, options?: unknown) => Promise<void>>();
		prompt.mockImplementationOnce(async (_text, options) => {
			(options as { preflightResult?: (success: boolean) => void })?.preflightResult?.(false);
			throw new ContextAdmissionDeniedError("blocked while steering");
		});
		prompt.mockImplementationOnce(async (_text, options) => {
			(options as { preflightResult?: (success: boolean) => void })?.preflightResult?.(true);
		});
		const context: EditorSubmitContext = {
			...submissionContext(),
			defaultEditor: {},
			editor: { setText: vi.fn(), addToHistory: vi.fn() },
			session: { isCompacting: false, isStreaming: true, isBashRunning: false, prompt },
			flushPendingBashComponents: vi.fn(),
			handleSubmission: prototype.handleSubmission,
			pendingUserInputs: [],
		};
		prototype.setupEditorSubmitHandler.call(context);

		await expect(context.defaultEditor.onSubmit?.("blocked steer")).resolves.toBeUndefined();
		await expect(context.defaultEditor.onSubmit?.("allowed steer")).resolves.toBeUndefined();

		expect(prompt).toHaveBeenNthCalledWith(
			1,
			"blocked steer",
			expect.objectContaining({ streamingBehavior: "steer" }),
		);
		expect(prompt).toHaveBeenNthCalledWith(
			2,
			"allowed steer",
			expect.objectContaining({ streamingBehavior: "steer" }),
		);
		expect(context.editor.setText).not.toHaveBeenCalledWith("blocked steer");
		expect(context.editor.addToHistory).not.toHaveBeenCalledWith("blocked steer");
		expect(context.editor.addToHistory).toHaveBeenCalledWith("allowed steer");
		expect(context.showError).toHaveBeenCalledWith("Message denied by context admission: blocked while steering");
	});

	it("makes a rejected follow-up recoverable through the same helper", async () => {
		const base = submissionContext();
		const context: FollowUpContext = {
			...base,
			editor: {
				setText: vi.fn(),
				getText: () => "blocked follow-up",
				addToHistory: vi.fn(),
			},
			session: {
				isCompacting: false,
				isStreaming: true,
				prompt: vi.fn(async (_text, options) => {
					(options as { preflightResult?: (success: boolean) => void })?.preflightResult?.(false);
					throw new ContextAdmissionDeniedError("blocked while following up");
				}),
			},
			queueCompactionMessage: vi.fn(),
			isExtensionCommand: vi.fn(() => false),
			updatePendingMessagesDisplay: vi.fn(),
			ui: { requestRender: vi.fn() },
		};

		await prototype.handleSubmission.call(context, () => prototype.handleFollowUp.call(context), "blocked follow-up");

		expect(context.editor.setText).not.toHaveBeenCalledWith("blocked follow-up");
		expect(context.editor.addToHistory).not.toHaveBeenCalledWith("blocked follow-up");
		expect(context.showError).toHaveBeenCalledWith("Message denied by context admission: blocked while following up");
	});
});
