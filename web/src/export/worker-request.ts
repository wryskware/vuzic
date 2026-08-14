import { readFile, stat } from 'node:fs/promises';
import { extname, isAbsolute } from 'node:path';

import {
  EXPORT_PROFILES,
  MAX_RECIPE_JSON_CHARS,
  liftExportRecipe,
  validateExportRecipe,
  type ExportProfile,
  type ExportRecipe,
} from '../runtime/recipe.ts';

export const EXPORT_WORKER_REQUEST_VERSION = 1 as const;
export const MAX_EXPORT_RANGE_SECONDS = 6 * 60 * 60;
export const MAX_WORKER_REQUEST_JSON_CHARS = MAX_RECIPE_JSON_CHARS + 64 * 1024;

export interface TimelineFilePaths {
  readonly jsonPath: string;
  readonly binaryPath: string;
}

export interface ExportWorkerRequestV1 {
  readonly version: typeof EXPORT_WORKER_REQUEST_VERSION;
  readonly recipe: ExportRecipe;
  /** Child-process settings are selected by the server, never by the browser. */
  readonly runtime: {
    readonly ffmpegExecutable: string;
    readonly workingDirectory: string;
  };
  /** Paths are resolved and authorized by the server before this file is written. */
  readonly timeline: TimelineFilePaths;
  readonly audioPath?: string;
  /** The worker owns `.partial` creation; this is the final server-selected path. */
  readonly output: {
    readonly path: string;
    readonly profile: ExportProfile;
  };
  readonly range: {
    readonly startSeconds: number;
    readonly durationSeconds: number;
  };
}

export type ExportWorkerRequest = ExportWorkerRequestV1;

function fail(path: string, message: string): never {
  throw new Error(`export worker request: ${path} ${message}`);
}

function object(value: unknown, path: string): Record<string, unknown> {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) {
    fail(path, 'must be an object');
  }
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) fail(path, 'must be a plain object');
  return value as Record<string, unknown>;
}

function keys(
  value: Record<string, unknown>,
  allowed: readonly string[],
  required: readonly string[],
  path: string,
): void {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) fail(`${path}.${key}`, 'is not supported by this schema version');
  }
  for (const key of required) {
    if (!Object.hasOwn(value, key)) fail(`${path}.${key}`, 'is required');
  }
}

function absolutePath(value: unknown, path: string, extension?: string): string {
  if (typeof value !== 'string' || value.length === 0 || value.length > 32_768) {
    fail(path, 'must be a non-empty path of at most 32768 characters');
  }
  if (value.includes('\0')) fail(path, 'must not contain a null byte');
  if (!isAbsolute(value)) fail(path, 'must be an absolute native path selected by the server');
  if (extension !== undefined && extname(value).toLowerCase() !== extension) {
    fail(path, `must end in ${extension}`);
  }
  return value;
}

function finite(value: unknown, path: string, min: number, max: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
    fail(path, `must be a finite number in ${min}..${max}`);
  }
  return value;
}

/** Validate the complete immutable Python-to-Node request-file contract. */
export function validateExportWorkerRequest(value: unknown): asserts value is ExportWorkerRequest {
  const request = object(value, '$');
  keys(
    request,
    ['version', 'recipe', 'runtime', 'timeline', 'audioPath', 'output', 'range'],
    ['version', 'recipe', 'runtime', 'timeline', 'output', 'range'],
    '$',
  );
  if (request['version'] !== EXPORT_WORKER_REQUEST_VERSION) {
    fail('$.version', `has unsupported version ${String(request['version'])}`);
  }
  try {
    // The headless replay path is exactly where an old sidecar shows up, so the
    // v3 → v4 palette lift happens here too. Assigning back into the request is
    // safe: the caller just parsed this object from JSON and nothing else holds
    // it, and the alternative — rejecting every export captured before palette
    // v2 — throws away recipes whose meaning is completely preserved.
    request['recipe'] = liftExportRecipe(request['recipe']);
    validateExportRecipe(request['recipe']);
  } catch (error) {
    fail('$.recipe', `is invalid (${(error as Error).message})`);
  }

  const runtime = object(request['runtime'], '$.runtime');
  keys(
    runtime,
    ['ffmpegExecutable', 'workingDirectory'],
    ['ffmpegExecutable', 'workingDirectory'],
    '$.runtime',
  );
  absolutePath(runtime['ffmpegExecutable'], '$.runtime.ffmpegExecutable');
  absolutePath(runtime['workingDirectory'], '$.runtime.workingDirectory');

  const timeline = object(request['timeline'], '$.timeline');
  keys(timeline, ['jsonPath', 'binaryPath'], ['jsonPath', 'binaryPath'], '$.timeline');
  const jsonPath = absolutePath(timeline['jsonPath'], '$.timeline.jsonPath', '.json');
  const binaryPath = absolutePath(timeline['binaryPath'], '$.timeline.binaryPath', '.bin');
  if (jsonPath === binaryPath) fail('$.timeline', 'JSON and binary paths must be different');

  if (Object.hasOwn(request, 'audioPath')) {
    absolutePath(request['audioPath'], '$.audioPath');
  }

  const output = object(request['output'], '$.output');
  keys(output, ['path', 'profile'], ['path', 'profile'], '$.output');
  absolutePath(output['path'], '$.output.path', '.mp4');
  if (!EXPORT_PROFILES.includes(output['profile'] as never)) {
    fail('$.output.profile', 'is unsupported');
  }
  const recipe = request['recipe'];
  if (output['profile'] !== recipe.output.profile) {
    fail('$.output.profile', 'must match $.recipe.output.profile');
  }

  const range = object(request['range'], '$.range');
  keys(range, ['startSeconds', 'durationSeconds'], ['startSeconds', 'durationSeconds'], '$.range');
  const start = finite(range['startSeconds'], '$.range.startSeconds', 0, MAX_EXPORT_RANGE_SECONDS);
  const duration = finite(
    range['durationSeconds'],
    '$.range.durationSeconds',
    Number.MIN_VALUE,
    MAX_EXPORT_RANGE_SECONDS,
  );
  if (start + duration > MAX_EXPORT_RANGE_SECONDS) {
    fail('$.range', `must end no later than ${MAX_EXPORT_RANGE_SECONDS} seconds`);
  }
}

export function parseExportWorkerRequest(text: string): ExportWorkerRequest {
  if (text.length > MAX_WORKER_REQUEST_JSON_CHARS) {
    fail('$', `exceeds ${MAX_WORKER_REQUEST_JSON_CHARS} serialized characters`);
  }
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    fail('$', `is not valid JSON (${(error as Error).message})`);
  }
  validateExportWorkerRequest(value);
  return value;
}

/** Read exactly one bounded request file; the worker never watches it for changes. */
export async function loadExportWorkerRequest(requestPath: string): Promise<ExportWorkerRequest> {
  absolutePath(requestPath, 'requestPath', '.json');
  const info = await stat(requestPath);
  if (!info.isFile()) fail('requestPath', 'must identify a regular file');
  if (info.size > MAX_WORKER_REQUEST_JSON_CHARS) {
    fail('requestPath', `exceeds ${MAX_WORKER_REQUEST_JSON_CHARS} bytes`);
  }
  return parseExportWorkerRequest(await readFile(requestPath, 'utf8'));
}
