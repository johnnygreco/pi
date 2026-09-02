import type { AssistantMessage, Context, Model, UserMessage } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import {
	admitModelContext,
	applyUserResult,
	buildProspectiveRequest,
	ManagedAdmissionError,
} from "../src/core/admission.ts";
import type { ModelRequestAdmissionEvent, ModelRequestAdmissionResult } from "../src/core/extensions/index.ts";

const model: Model<"openai-completions"> = {
	id: "fixture-model",
	name: "Fixture",
	api: "openai-completions",
	provider: "fixture-provider",
	baseUrl: "https://provider.invalid/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128_000,
	maxTokens: 4096,
};

function candidate(text: string): UserMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp: 1 };
}

function context(messages: Context["messages"] = [candidate("safe")]): Context {
	return { systemPrompt: "system", messages, tools: [] };
}

function runner(result: ModelRequestAdmissionResult) {
	return {
		emitModelRequestAdmission: async (_event: ModelRequestAdmissionEvent) => result,
	};
}

describe("managed admission", () => {
	it("uses provider replay normalization and preserves candidate identity by user order", () => {
		const usage = {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
			totalTokens: 0,
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
		};
		const errored: AssistantMessage = {
			role: "assistant",
			content: [{ type: "text", text: "partial" }],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage,
			stopReason: "error",
			errorMessage: "untrusted detail",
			timestamp: 2,
		};
		const first = candidate("first");
		const next = { ...candidate("next"), timestamp: first.timestamp };
		const prospective = buildProspectiveRequest(model, context([first, errored, next]), "agentInitial", 2);

		expect(prospective.messages).toEqual([
			{ role: "user", content: "first", toolCalls: [] },
			{ role: "user", content: "next", toolCalls: [] },
		]);
		expect(prospective.candidateIndex).toBe(1);
	});

	it("returns the exact admitted context and bound transport values", async () => {
		const requestContext = context();
		const receipt = new TextEncoder().encode("eg1.fixture.receipt");
		const admitted = await admitModelContext(
			runner({ action: "replace", systemPrompt: "safe system", receipt }),
			model,
			requestContext,
			"agentInitial",
		);

		expect(admitted.context).toEqual({ ...requestContext, systemPrompt: "safe system" });
		expect(admitted).toMatchObject({ receipt, temperature: 0, maxTokens: 4096, toolChoice: "auto" });
	});

	it.each([
		[undefined, "invalid_admission_response"],
		[null, "invalid_admission_response"],
		[{}, "invalid_admission_response"],
		[{ action: "unknown" }, "invalid_admission_response"],
		[{ action: "allow" }, "invalid_receipt"],
	] as const)("fails closed on malformed model result %#", async (result, reasonCode) => {
		const invalidRunner = {
			emitModelRequestAdmission: async () => result as never,
		};
		await expect(admitModelContext(invalidRunner, model, context(), "agentInitial")).rejects.toMatchObject({
			name: "ManagedAdmissionError",
			reasonCode,
		});
	});

	it("normalizes handler failures and cancellation", async () => {
		await expect(
			admitModelContext(
				{ emitModelRequestAdmission: async () => Promise.reject(new Error("bridge detail")) },
				model,
				context(),
				"agentInitial",
			),
		).rejects.toMatchObject({ reasonCode: "admission_unavailable" });

		const controller = new AbortController();
		controller.abort();
		await expect(
			admitModelContext(
				runner({ action: "allow", receipt: new Uint8Array([1]) }),
				model,
				context(),
				"retry",
				controller.signal,
			),
		).rejects.toMatchObject({ reasonCode: "admission_cancelled" });
	});

	it("admits each retry as a distinct request", async () => {
		const events: ModelRequestAdmissionEvent[] = [];
		const retryRunner = {
			emitModelRequestAdmission: async (event: ModelRequestAdmissionEvent) => {
				events.push(event);
				return { action: "allow" as const, receipt: new TextEncoder().encode(`receipt-${events.length}`) };
			},
		};
		await admitModelContext(retryRunner, model, context(), "retry");
		await admitModelContext(retryRunner, model, context(), "retry");

		expect(events.map((event) => event.requestKind)).toEqual(["retry", "retry"]);
		expect(events[0]?.modelRequestId).not.toBe(events[1]?.modelRequestId);
	});

	it("allows only a same-identity early user replacement", () => {
		const original = candidate("unsafe");
		const replacement = candidate("safe");
		expect(applyUserResult(original, { action: "replace", message: replacement })).toBe(replacement);
		expect(() => applyUserResult(original, null)).toThrow(ManagedAdmissionError);
		expect(() => applyUserResult(original, { action: "replace", message: { ...replacement, timestamp: 2 } })).toThrow(
			ManagedAdmissionError,
		);
	});
});
