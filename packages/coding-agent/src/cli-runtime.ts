import { APP_NAME } from "./config.ts";
import { configureHttpDispatcher } from "./core/http-dispatcher.ts";
import { type MainOptions, main } from "./main.ts";

export type CliRuntimeOptions = MainOptions;

/** Run Pi's standard CLI with optional trusted runtime integrations. */
export function runCli(args: string[], options?: CliRuntimeOptions): Promise<void> {
	process.title = APP_NAME;
	process.env.PI_CODING_AGENT = "true";
	process.env.AI_AGENT = "pi";
	process.emitWarning = (() => {}) as typeof process.emitWarning;

	// Configure undici before provider SDKs issue requests. Runtime settings are
	// applied once SettingsManager has loaded global/project settings.
	configureHttpDispatcher();
	return main(args, options);
}
