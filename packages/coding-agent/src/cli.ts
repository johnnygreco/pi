#!/usr/bin/env node
/**
 * CLI entry point for the refactored coding agent.
 * Uses main.ts with AgentSession and new mode modules.
 *
 * Test with: npx tsx src/cli-new.ts [args...]
 */
import { APP_NAME } from "./config.ts";
import { configureHttpDispatcher } from "./core/http-dispatcher.ts";
import { createOpenShellContextAdmission } from "./core/openshell-context-admission.ts";
import { main } from "./main.ts";

process.title = APP_NAME;
process.env.PI_CODING_AGENT = "true";
process.env.AI_AGENT = "pi";
process.emitWarning = (() => {}) as typeof process.emitWarning;

// Configure undici's global dispatcher before provider SDKs issue requests.
// Runtime settings are applied once SettingsManager has loaded global/project settings.
configureHttpDispatcher();

const openshellAdmissionRequired = process.env.PI_OPENSHELL_CONTEXT_ADMISSION === "1";
const openshellBridgeUrl = process.env.OPENSHELL_AGENT_CONVERSATION_URL;
if (openshellAdmissionRequired && !openshellBridgeUrl) {
	throw new Error("OPENSHELL_AGENT_CONVERSATION_URL is required when PI_OPENSHELL_CONTEXT_ADMISSION=1");
}

main(
	process.argv.slice(2),
	openshellAdmissionRequired
		? {
				createContextAdmission: (sessionManager) =>
					createOpenShellContextAdmission(openshellBridgeUrl!, () => sessionManager.getSessionId()),
			}
		: undefined,
);
