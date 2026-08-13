import { readFile } from 'node:fs/promises';

import type { Timeline } from '../timeline/types.ts';
import { createTimeline } from '../timeline/validate.ts';
import type { TimelineFilePaths } from './worker-request.ts';

/** Filesystem transport adapter for the same parser used by browser fetches. */
export async function loadTimelineFromFiles(paths: TimelineFilePaths): Promise<Timeline> {
  const [manifestText, binary] = await Promise.all([
    readFile(paths.jsonPath, 'utf8'),
    readFile(paths.binaryPath),
  ]);

  let manifest: unknown;
  try {
    manifest = JSON.parse(manifestText);
  } catch (error) {
    throw new Error(`timeline.json: invalid JSON (${(error as Error).message})`);
  }
  return createTimeline(manifest, binary);
}
