import type { Context, ToolResultMessage, UserMessage } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";

import { createOpenShellContextAdmission } from "../src/core/openshell-context-admission.ts";

const HANDLE_HEADER = "x-openshell-agent-admission-handle";

function user(text: string, timestamp: number): UserMessage {
	return { role: "user", content: [{ type: "text", text }], timestamp };
}

function toolResult(text: string, isError = false): ToolResultMessage {
	return {
		role: "toolResult",
		toolCallId: "call-1",
		toolName: "bash",
		content: [{ type: "text", text }],
		isError,
		timestamp: 2,
	};
}

async function admittedContext(
	admission: ReturnType<typeof createOpenShellContextAdmission>,
	context: Context,
): Promise<Context> {
	const result = await admission.admitProviderContext(context);
	expect(result.action).toBe("allow");
	return result.action === "allow" ? (result.context ?? context) : context;
}

describe("OpenShell context admission", () => {
	it("selects the handle for the exact provider context", async () => {
		const admission = createOpenShellContextAdmission(
			"http://bridge.test/admit",
			() => "session-123",
			async (_url, init) => {
				const request = JSON.parse(String(init?.body));
				const envelope = JSON.parse(new TextDecoder().decode(new Uint8Array(request.request_body)));
				return new Response(JSON.stringify({ decision: "allow", handle: `handle:${envelope.text}` }));
			},
		);
		const current = user("current", 1);
		const queued = user("queued", 2);

		expect((await admission.admitUserMessage(current, { source: "interactive" })).action).toBe("allow");
		expect((await admission.admitUserMessage(queued, { source: "interactive" })).action).toBe("allow");

		const currentHeaders = await admission.transformProviderHeaders(
			{},
			await admittedContext(admission, { messages: [current], tools: [] }),
		);
		const queuedHeaders = await admission.transformProviderHeaders(
			{},
			await admittedContext(admission, { messages: [current, queued], tools: [] }),
		);

		expect(currentHeaders[HANDLE_HEADER]).toBe("handle:current");
		expect(queuedHeaders[HANDLE_HEADER]).toBe("handle:queued");
	});

	it("uses an admitted replacement for the outbound handle", async () => {
		const replacement = new TextEncoder().encode(
			JSON.stringify({ schema_version: "openshell.pi-input.v1", text: "[REDACTED]" }),
		);
		const admission = createOpenShellContextAdmission(
			"http://bridge.test/admit",
			() => "session-123",
			async () =>
				new Response(
					JSON.stringify({ decision: "allow", handle: "replacement-handle", replacement_body: [...replacement] }),
				),
		);
		const admitted = await admission.admitUserMessage(user("secret", 1), { source: "interactive" });

		expect(admitted.action).toBe("allow");
		if (admitted.action !== "allow" || !admitted.message) throw new Error("expected replacement");
		const context = await admittedContext(admission, { messages: [admitted.message], tools: [] });
		const headers = await admission.transformProviderHeaders({}, context);

		expect(admitted.message.content).toEqual([{ type: "text", text: "[REDACTED]" }]);
		expect(headers[HANDLE_HEADER]).toBe("replacement-handle");
	});

	it("attests failed tool results", async () => {
		const hooks: string[] = [];
		const admission = createOpenShellContextAdmission(
			"http://bridge.test/admit",
			() => "session-123",
			async (_url, init) => {
				const request = JSON.parse(String(init?.body));
				hooks.push(request.hook);
				return new Response(JSON.stringify({ decision: "allow", handle: `handle:${request.hook}` }));
			},
		);
		const prompt = user("run command", 1);
		const failed = toolResult("Command exited with code 2", true);

		await admission.admitUserMessage(prompt, { source: "interactive" });
		await admission.admitToolResult(failed);
		const headers = await admission.transformProviderHeaders(
			{},
			await admittedContext(admission, { messages: [prompt, failed], tools: [] }),
		);

		expect(headers[HANDLE_HEADER]).toBe("handle:tool_result_admission");
		expect(hooks).toEqual(["rendered_prompt_admission", "tool_result_admission", "tool_result_admission"]);
	});

	it("fails closed when provider-only context is denied", async () => {
		const admission = createOpenShellContextAdmission(
			"http://bridge.test/admit",
			() => "session-123",
			async () => new Response(JSON.stringify({ decision: "deny", reason_code: "policy_denied" })),
		);

		await expect(admission.admitProviderContext({ messages: [user("summary", 1)], tools: [] })).resolves.toEqual({
			action: "deny",
			reason: "OpenShell denied this context addition (policy_denied)",
		});
	});
});
