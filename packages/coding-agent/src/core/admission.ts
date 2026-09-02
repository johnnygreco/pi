import type { Context, Message, Model, ToolCall, UserMessage } from "@earendil-works/pi-ai";
import { normalizeOpenAICompletionsMessages } from "@earendil-works/pi-ai/api/openai-completions";
import { clampMaxTokensToContext } from "@earendil-works/pi-ai/api/simple-options";
import type {
	ExtensionRunner,
	PiProspectiveMessageV1,
	PiProspectiveRequestV1,
	PiRequestKind,
} from "./extensions/index.ts";

const MAX_RECEIPT_BYTES = 16 * 1024;
const REASON_CODE_PATTERN = /^[a-z][a-z0-9_]{0,63}$/;

export class ManagedAdmissionError extends Error {
	readonly reasonCode: string;

	constructor(reasonCode: string) {
		super(`Managed admission denied (${reasonCode})`);
		this.name = "ManagedAdmissionError";
		this.reasonCode = reasonCode;
	}
}

export interface AdmittedRequest {
	context: Context;
	receipt: Uint8Array;
	temperature: number;
	maxTokens: number;
	toolChoice: "auto";
}

export interface ManagedAdmissionSessionState {
	nextRequestKind?: PiRequestKind;
	onRequestAdmitted?: () => void;
}

type ModelAdmissionRunner = Pick<ExtensionRunner, "emitModelRequestAdmission">;

export function buildProspectiveRequest(
	model: Model<any>,
	context: Context,
	requestKind: PiRequestKind,
	candidateIndex?: number,
): PiProspectiveRequestV1 {
	if (model.api !== "openai-completions") throw new ManagedAdmissionError("unsupported_model_api");
	const normalizedMessages = normalizeOpenAICompletionsMessages(
		model as Model<"openai-completions">,
		context.messages,
	);
	const normalizedCandidateIndex = findNormalizedCandidateIndex(context.messages, normalizedMessages, candidateIndex);
	return {
		schemaVersion: "pi.prospective-request.v1",
		model: { provider: model.provider, id: model.id, api: model.api },
		systemPrompt: context.systemPrompt ?? "",
		messages: normalizedMessages.map(convertMessage),
		tools: (context.tools ?? []).map((tool) => ({
			name: tool.name,
			description: tool.description,
			inputSchema: tool.parameters as Record<string, unknown>,
		})),
		toolChoice: { mode: "auto" },
		generation: {
			temperature: 0,
			maxTokens: clampMaxTokensToContext(model, context, model.maxTokens),
		},
		candidateIndex: normalizedCandidateIndex,
		requestKind,
	};
}

export async function admitModelContext(
	runner: ModelAdmissionRunner,
	model: Model<any>,
	context: Context,
	requestKind: PiRequestKind,
	signal?: AbortSignal,
): Promise<AdmittedRequest> {
	const request = buildProspectiveRequest(model, context, requestKind);
	let result: unknown;
	try {
		result = await runner.emitModelRequestAdmission({
			type: "model_request_admission",
			modelRequestId: crypto.randomUUID(),
			requestKind,
			request,
			signal,
		});
	} catch {
		throw new ManagedAdmissionError("admission_unavailable");
	}
	if (signal?.aborted) throw new ManagedAdmissionError("admission_cancelled");
	return applyModelResult(context, request, result);
}

export function applyUserResult(message: UserMessage, result: unknown): UserMessage {
	if (!isRecord(result) || typeof result.action !== "string") {
		throw new ManagedAdmissionError("invalid_admission_response");
	}
	if (result.action === "deny") {
		throw new ManagedAdmissionError(validateReasonCode(result.reasonCode));
	}
	if (result.action === "allow") return message;
	if (result.action !== "replace" || !isUserMessage(result.message)) {
		throw new ManagedAdmissionError("invalid_admission_response");
	}
	validateUserReplacement(message, result.message);
	return result.message;
}

function applyModelResult(context: Context, request: PiProspectiveRequestV1, result: unknown): AdmittedRequest {
	if (!isRecord(result) || typeof result.action !== "string") {
		throw new ManagedAdmissionError("invalid_admission_response");
	}
	if (result.action === "deny") {
		throw new ManagedAdmissionError(validateReasonCode(result.reasonCode));
	}
	if (result.action !== "allow" && result.action !== "replace") {
		throw new ManagedAdmissionError("invalid_admission_response");
	}
	if (!(result.receipt instanceof Uint8Array)) {
		throw new ManagedAdmissionError("invalid_receipt");
	}
	validateReceipt(result.receipt);
	let admittedContext = context;
	if (result.action === "replace") {
		if (typeof result.systemPrompt !== "string") {
			throw new ManagedAdmissionError("invalid_replacement");
		}
		admittedContext = { ...context, systemPrompt: result.systemPrompt };
	}
	return {
		context: admittedContext,
		receipt: result.receipt,
		temperature: request.generation.temperature,
		maxTokens: request.generation.maxTokens,
		toolChoice: "auto",
	};
}

function findNormalizedCandidateIndex(
	messages: Message[],
	normalizedMessages: Message[],
	candidateIndex: number | undefined,
): number | undefined {
	if (candidateIndex === undefined) return undefined;
	const candidate = messages[candidateIndex];
	if (!candidate || candidate.role !== "user") throw new ManagedAdmissionError("candidate_not_model_visible");
	const userOrdinal = messages.slice(0, candidateIndex + 1).filter((message) => message.role === "user").length - 1;
	let currentUserOrdinal = 0;
	for (const [index, message] of normalizedMessages.entries()) {
		if (message.role !== "user") continue;
		if (currentUserOrdinal === userOrdinal) return index;
		currentUserOrdinal++;
	}
	throw new ManagedAdmissionError("candidate_not_model_visible");
}

function convertMessage(message: Message): PiProspectiveMessageV1 {
	if (message.role === "user") return { role: "user", content: textContent(message.content), toolCalls: [] };
	if (message.role === "toolResult") {
		const content = textContent(message.content);
		return {
			role: "tool",
			content: content.length > 0 ? content : "(no tool output)",
			toolCallId: message.toolCallId,
			toolCalls: [],
		};
	}
	const textParts = message.content.filter((part) => part.type === "text");
	if (message.content.some((part) => part.type === "thinking") || textParts.length > 1) {
		throw new ManagedAdmissionError("unsupported_message_content");
	}
	const toolCalls = message.content.filter((part): part is ToolCall => part.type === "toolCall");
	const text = textParts[0]?.text;
	if (toolCalls.length === 0 && (text === undefined || text.trim().length === 0)) {
		throw new ManagedAdmissionError("unsupported_message_content");
	}
	return {
		role: "assistant",
		content: text !== undefined && text.trim().length > 0 ? text : null,
		toolCalls: toolCalls.map((call) => ({
			id: call.id,
			name: call.name,
			arguments: JSON.stringify(call.arguments),
		})),
	};
}

function textContent(content: UserMessage["content"]): string {
	if (typeof content === "string") return content;
	if (content.length !== 1 || content[0]?.type !== "text") {
		throw new ManagedAdmissionError("unsupported_message_content");
	}
	return content[0].text;
}

function validateUserReplacement(original: UserMessage, replacement: UserMessage): void {
	if (replacement.timestamp !== original.timestamp) throw new ManagedAdmissionError("invalid_replacement");
	textContent(replacement.content);
}

function validateReceipt(receipt: Uint8Array): void {
	if (
		receipt.byteLength === 0 ||
		receipt.byteLength > MAX_RECEIPT_BYTES ||
		receipt.some((byte) => byte < 0x21 || byte > 0x7e)
	) {
		throw new ManagedAdmissionError("invalid_receipt");
	}
}

function validateReasonCode(value: unknown): string {
	return typeof value === "string" && REASON_CODE_PATTERN.test(value) ? value : "invalid_admission_response";
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return value !== null && typeof value === "object";
}

function isUserMessage(value: unknown): value is UserMessage {
	return isRecord(value) && value.role === "user" && typeof value.timestamp === "number" && "content" in value;
}
