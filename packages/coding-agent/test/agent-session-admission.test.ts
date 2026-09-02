import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type AssistantMessage,
	createAssistantMessageEventStream,
	type Model,
	type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import { ManagedAdmissionError } from "../src/core/admission.ts";
import { AuthStorage } from "../src/core/auth-storage.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { createModelRegistry, getModelRuntime } from "./model-runtime-test-utils.ts";
import { createTestExtensionsResult, createTestResourceLoader } from "./utilities.ts";

const model: Model<"openai-completions"> = {
	id: "admission-model",
	name: "Admission Model",
	api: "openai-completions",
	provider: "admission-provider",
	baseUrl: "https://provider.invalid/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 128_000,
	maxTokens: 4096,
};

describe("AgentSession managed admission", () => {
	const cleanups: Array<() => void> = [];

	afterEach(() => {
		while (cleanups.length > 0) cleanups.pop()?.();
	});

	it("denies a rendered user submission before memory, persistence, and provider work", async () => {
		const cwd = mkdtempSync(join(tmpdir(), "pi-admission-"));
		cleanups.push(() => rmSync(cwd, { recursive: true, force: true }));
		const authStorage = AuthStorage.inMemory();
		await authStorage.modify(model.provider, async () => ({ type: "api_key", key: "fixture-key" }));
		const registry = await createModelRegistry(authStorage, join(cwd, "models.json"));
		let providerCalls = 0;
		let providerOptions: SimpleStreamOptions | undefined;
		let releaseProvider: (() => void) | undefined;
		let providerStartedResolve = () => {};
		const providerStarted = new Promise<void>((resolve) => {
			providerStartedResolve = resolve;
		});
		registry.registerProvider(model.provider, {
			api: model.api,
			apiKey: "fixture-key",
			models: [model],
			streamSimple: (_model, requestContext, options) => {
				providerCalls++;
				providerOptions = options;
				const stream = createAssistantMessageEventStream();
				const response: AssistantMessage = {
					role: "assistant",
					content: [{ type: "text", text: "ok" }],
					api: model.api,
					provider: model.provider,
					model: model.id,
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
				const finish = () => stream.end(response);
				if (
					requestContext.messages.some(
						(message) =>
							message.role === "user" &&
							typeof message.content !== "string" &&
							message.content[0]?.type === "text" &&
							message.content[0].text === "slow",
					)
				) {
					releaseProvider = finish;
					providerStartedResolve();
				} else {
					finish();
				}
				return stream;
			},
		});
		cleanups.push(() => registry.unregisterProvider(model.provider));
		const missingHandlers = await createTestExtensionsResult([], cwd);
		await expect(
			createAgentSession({
				cwd,
				model,
				modelRuntime: getModelRuntime(registry),
				settingsManager: SettingsManager.inMemory(),
				sessionManager: SessionManager.inMemory(cwd),
				resourceLoader: createTestResourceLoader({ extensionsResult: missingHandlers }),
				managedAdmission: true,
			}),
		).rejects.toThrow("exactly one user_message_admission handler; found 0");
		const payloadMutatingExtensions = await createTestExtensionsResult(
			[
				(pi) => {
					pi.on("user_message_admission", () => ({ action: "allow" }));
					pi.on("model_request_admission", () => ({
						action: "allow",
						receipt: new Uint8Array([1]),
					}));
					pi.on("before_provider_request", (event) => event);
				},
			],
			cwd,
		);
		await expect(
			createAgentSession({
				cwd,
				model,
				modelRuntime: getModelRuntime(registry),
				settingsManager: SettingsManager.inMemory(),
				sessionManager: SessionManager.inMemory(cwd),
				resourceLoader: createTestResourceLoader({ extensionsResult: payloadMutatingExtensions }),
				managedAdmission: true,
			}),
		).rejects.toThrow("Managed admission is incompatible with before_provider_request handlers");
		const extensionsResult = await createTestExtensionsResult(
			[
				(pi) => {
					pi.on("before_agent_start", (event) =>
						event.prompt === "blocked at model boundary"
							? {
									systemPrompt: "speculative system prompt",
									messages: [{ customType: "speculative", content: "not durable", display: false }],
								}
							: undefined,
					);
					pi.on("user_message_admission", (event) => {
						const rendered = event.prospectiveRequest.messages[event.candidateIndex]?.content;
						if (rendered === "early bridge failed") throw new Error("simulated bridge timeout");
						if (rendered === "malformed early") return undefined as never;
						return rendered === "blocked rendered text"
							? { action: "deny", reasonCode: "policy_denied" }
							: { action: "allow" };
					});
					pi.on("model_request_admission", (event) => {
						if (event.request.messages.some((message) => message.content === "malformed model")) {
							return null as never;
						}
						if (
							event.request.messages.some(
								(message) => message.role === "user" && message.content === "bridge failed",
							)
						) {
							throw new Error("simulated malformed bridge response");
						}
						const blocked = event.request.messages.some(
							(message) => message.role === "user" && message.content === "blocked at model boundary",
						);
						return blocked
							? { action: "deny", reasonCode: "model_policy_denied" }
							: {
									action: "allow",
									receipt: new TextEncoder().encode("eg1.fixture.receipt"),
								};
					});
				},
			],
			cwd,
		);
		const sessionManager = SessionManager.inMemory(cwd);
		const { session } = await createAgentSession({
			cwd,
			model,
			modelRuntime: getModelRuntime(registry),
			settingsManager: SettingsManager.inMemory(),
			sessionManager,
			resourceLoader: createTestResourceLoader({ extensionsResult }),
			managedAdmission: true,
		});
		cleanups.push(() => session.dispose());
		const baseSystemPrompt = session.systemPrompt;

		let deniedPreflight: boolean | undefined;
		await expect(
			session.prompt("blocked rendered text", {
				preflightResult: (result) => {
					deniedPreflight = result;
				},
			}),
		).rejects.toBeInstanceOf(ManagedAdmissionError);
		expect(deniedPreflight).toBe(false);
		expect(session.messages).toEqual([]);
		expect(sessionManager.getBranch().filter((entry) => entry.type === "message")).toEqual([]);
		expect(providerCalls).toBe(0);

		await expect(session.prompt("malformed early")).rejects.toMatchObject({
			name: "ManagedAdmissionError",
			reasonCode: "invalid_admission_response",
		});
		await expect(session.prompt("malformed model")).rejects.toMatchObject({
			name: "ManagedAdmissionError",
			reasonCode: "invalid_admission_response",
		});
		expect(session.messages).toEqual([]);
		expect(providerCalls).toBe(0);

		await expect(session.prompt("early bridge failed")).rejects.toMatchObject({
			name: "ManagedAdmissionError",
			reasonCode: "admission_unavailable",
		});
		expect(session.messages).toEqual([]);
		expect(sessionManager.getBranch().filter((entry) => entry.type === "message")).toEqual([]);
		expect(providerCalls).toBe(0);

		let modelDeniedPreflight: boolean | undefined;
		await expect(
			session.prompt("blocked at model boundary", {
				preflightResult: (result) => {
					modelDeniedPreflight = result;
				},
			}),
		).rejects.toBeInstanceOf(ManagedAdmissionError);
		expect(modelDeniedPreflight).toBe(false);
		expect(session.messages).toEqual([]);
		expect(session.systemPrompt).toBe(baseSystemPrompt);
		expect(sessionManager.getBranch().filter((entry) => entry.type === "message")).toEqual([]);
		expect(providerCalls).toBe(0);

		await expect(session.prompt("bridge failed")).rejects.toMatchObject({
			name: "ManagedAdmissionError",
			reasonCode: "admission_unavailable",
		});
		expect(session.messages).toEqual([]);
		expect(session.systemPrompt).toBe(baseSystemPrompt);
		expect(sessionManager.getBranch().filter((entry) => entry.type === "message")).toEqual([]);
		expect(providerCalls).toBe(0);

		await session.prompt("safe rendered text");
		expect(providerCalls, JSON.stringify(session.messages)).toBe(1);
		expect(session.messages[0]).toMatchObject({ role: "user" });
		expect(providerOptions?.headers?.["x-openshell-middleware-egress-receipt"]).toBe("eg1.fixture.receipt");
		expect(providerOptions).toMatchObject({ temperature: 0, maxTokens: 4096, toolChoice: "auto", maxRetries: 0 });

		let preflight: boolean | undefined;
		let settled = false;
		const slowPrompt = session
			.prompt("slow", {
				preflightResult: (result) => {
					preflight = result;
				},
			})
			.then(() => {
				settled = true;
			});
		await providerStarted;
		expect(preflight).toBe(true);
		expect(settled).toBe(false);
		releaseProvider?.();
		await slowPrompt;
	});
});
