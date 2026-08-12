import { join } from "node:path";
import { afterEach, describe, expect, test } from "vitest";
import { resolveOpenShellAdmissionExtension } from "../src/main.ts";

const originalPackageDir = process.env.PI_PACKAGE_DIR;

afterEach(() => {
	if (originalPackageDir === undefined) delete process.env.PI_PACKAGE_DIR;
	else process.env.PI_PACKAGE_DIR = originalPackageDir;
});

describe("managed OpenShell admission", () => {
	test("loads the bundled extension from the active package", () => {
		process.env.PI_PACKAGE_DIR = "/opt/pi-coding-agent";

		expect(resolveOpenShellAdmissionExtension("http://127.0.0.1:8193", false)).toBe(
			join("/opt/pi-coding-agent", "examples", "extensions", "openshell-input-admission.ts"),
		);
	});

	test("rejects disabling extensions when the bridge is active", () => {
		expect(() => resolveOpenShellAdmissionExtension("http://127.0.0.1:8193", true)).toThrow(
			"--no-extensions cannot be used",
		);
	});

	test("does not load the extension outside managed OpenShell", () => {
		expect(resolveOpenShellAdmissionExtension(undefined, false)).toBeUndefined();
	});
});
