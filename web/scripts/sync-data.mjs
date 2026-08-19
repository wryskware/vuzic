import { copyFile, mkdir, readdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';
import { createHash } from 'node:crypto';

const run = promisify(execFile);

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, '../../data/timelines');
const dst = resolve(here, '../public/timelines');
const allowlistPath = resolve(here, '../../data/publish.json');

/**
 * Files the browser actually asks for. **An allowlist, not a skip list.**
 *
 * The previous skip list named the two known offenders (`embedding.*`, `plots/`)
 * and copied everything else, which is the wrong default for a directory the
 * analysis pipeline writes into: `bug-fix-rush` alone carried 116 MB of demucs
 * stems in `demix/` and 20 MB of spectrograms in `spec/` straight into `public/`
 * and therefore into `dist/`, because nobody had thought to name them. Under an
 * allowlist a new analysis artefact costs nothing until someone asks for it.
 *
 * `run.json` is deliberately absent — nothing in `web/src` reads it. The raw MuQ
 * `embedding.*` sidecar stays in `data/` as the training input for the distilled
 * mapping in plan.md's "Later"; it just does not ship.
 */
const WANTED = ['timeline.json', 'timeline.bin'];

/**
 * The published rendition of a track's audio.
 *
 * `audio.wav` is 30-52 MB per track: fine over localhost, indefensible over
 * CloudFront to a friends-and-family link. AAC in an MP4 container is what
 * `decodeAudioData` supports everywhere without qualification (Opus would save
 * more bytes and cost Safari), and `+faststart` puts the moov atom first so the
 * fetch is useful before it completes.
 *
 * Bundled tracks therefore serve `audio.m4a` while `terrarium-server` still
 * serves `audio.wav` from the same pipeline output. That difference is why the
 * file name rides the catalog entry instead of being a constant: see
 * `TrackEntry.audioFile`.
 */
const AUDIO_SRC = 'audio.wav';
const AUDIO_OUT = 'audio.m4a';
const AUDIO_BITRATE = '160k';

// A flag, not an env var: `LMT_PUBLISH=1 npm run ...` is not portable to
// PowerShell or cmd, and this repo is developed on Windows.
const publishing = process.argv.includes('--publish');

/**
 * The fallback track always ships. `main.ts` falls back to `synthetic` when a
 * requested track will not load, so a publish build without it has a dead
 * default path - and it is 700 KB with no audio, so it costs nothing.
 */
const ALWAYS = new Set(['synthetic']);

async function publishAllowlist() {
  const raw = await readFile(allowlistPath, 'utf8').catch(() => null);
  if (raw === null) {
    console.error(`sync-data: --publish given but no allowlist at ${allowlistPath}`);
    process.exit(1);
  }
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed.tracks) || parsed.tracks.some((t) => typeof t !== 'string')) {
    console.error('sync-data: data/publish.json needs a "tracks" array of strings');
    process.exit(1);
  }
  return new Set([...parsed.tracks, ...ALWAYS]);
}

/**
 * Transcode if the source is newer than the output, else reuse. Keyed on mtime
 * rather than content because a 40 MB hash per track per build buys nothing that
 * a timestamp does not.
 */
async function encodeAudio(from, to) {
  const [srcStat, outStat] = await Promise.all([stat(from), stat(to).catch(() => null)]);
  if (outStat && outStat.mtimeMs >= srcStat.mtimeMs) return true;
  try {
    await run('ffmpeg', ['-y', '-i', from, '-c:a', 'aac', '-b:a', AUDIO_BITRATE,
      '-movflags', '+faststart', to]);
    return true;
  } catch (err) {
    // Not fatal: the audio clock's click-track fallback is exactly the path a
    // bundled timeline with no audio already takes, and a missing ffmpeg must
    // not stop `npm run dev` from starting.
    console.warn(`sync-data: could not encode ${from} (${err.code ?? err.message}); ` +
      'track ships without audio. Install ffmpeg and re-run.');
    return false;
  }
}

const allow = publishing ? await publishAllowlist() : null;
if (allow) console.log(`sync-data: publish build - ${allow.size} allowlisted tracks`);

const entries = await readdir(src, { withFileTypes: true }).catch(() => {
  console.error(`sync-data: no source directory at ${src}`);
  process.exit(1);
});

// A publish build must not inherit a previous dev build's tracks: `public/` is
// not cleaned by vite, so without this the allowlist would only govern what gets
// *added*, and every track ever synced would ship anyway.
if (publishing) await rm(dst, { recursive: true, force: true });
await mkdir(dst, { recursive: true });

const index = [];
for (const e of entries) {
  if (!e.isDirectory() || e.name.startsWith('.')) continue;
  if (allow && !allow.has(e.name)) continue;

  const from = join(src, e.name);
  const into = join(dst, e.name);
  await mkdir(into, { recursive: true });
  for (const name of WANTED) {
    await copyFile(join(from, name), join(into, name)).catch(() => undefined);
  }

  const hasSourceAudio = await stat(join(from, AUDIO_SRC)).then(() => true, () => false);
  const hasAudio =
    hasSourceAudio && (await encodeAudio(join(from, AUDIO_SRC), join(into, AUDIO_OUT)));

  // The track picker needs to know what shipped, and the set of bundled tracks
  // is a property of this directory rather than of any list in the source - so
  // it is written here instead of being maintained by hand in TypeScript.
  // Malformed timelines are skipped rather than fatal: `sync-data` runs before
  // every `dev` and `build`, and a half-written analysis output must not stop
  // the dev server from starting.
  const manifestBytes = await readFile(join(from, 'timeline.json')).catch(() => null);
  const binaryBytes = await readFile(join(from, 'timeline.bin')).catch(() => null);
  const manifest =
    manifestBytes === null
      ? null
      : (() => {
          try {
            return JSON.parse(manifestBytes.toString('utf8'));
          } catch {
            return null;
          }
        })();
  if (manifest?.track && binaryBytes !== null) {
    index.push({
      id: e.name,
      title: String(manifest.track.id ?? e.name),
      duration: Number(manifest.track.duration) || 0,
      // Same content identity as FastAPI: exact manifest bytes followed by the
      // exact binary timeline bytes. An export may therefore prove that the
      // browser session and server worker will drive from identical inputs.
      // Deliberately unaffected by the audio rendition - the timeline is what
      // drives the simulation, and a re-encode is not a different track.
      version: createHash('sha256')
        .update(manifestBytes)
        .update(binaryBytes)
        .digest('hex')
        .slice(0, 16),
      hasAudio,
      ...(hasAudio ? { audio: AUDIO_OUT } : {}),
    });
  } else {
    console.warn(`sync-data: ${e.name} has no readable timeline pair; not listed`);
  }
  console.log(`sync-data: ${e.name}${hasAudio ? '' : ' (no audio)'}`);
}

index.sort((a, b) => a.title.localeCompare(b.title));
await writeFile(join(dst, 'index.json'), `${JSON.stringify(index, null, 2)}\n`);
console.log(`sync-data: index.json (${index.length} tracks)`);
