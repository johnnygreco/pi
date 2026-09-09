import { setupCli } from "./cli/setup.ts";
import { type MainOptions, main } from "./main.ts";

export type CliRuntimeOptions = MainOptions;

/** Run Pi's standard CLI with optional trusted runtime integrations. */
export function runCli(args: string[], options?: CliRuntimeOptions): Promise<void> {
	setupCli();
	return main(args, options);
}
