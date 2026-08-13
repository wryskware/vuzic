import assert from 'node:assert/strict';
import { after, test } from 'node:test';

const originalLocation = Object.getOwnPropertyDescriptor(globalThis, 'location');
const originalStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
const values = new Map<string, string>();

Object.defineProperty(globalThis, 'location', {
  configurable: true,
  value: { search: '' },
});
Object.defineProperty(globalThis, 'localStorage', {
  configurable: true,
  value: {
    getItem: (key: string): string | null => values.get(key) ?? null,
    setItem: (key: string, value: string): void => void values.set(key, value),
    removeItem: (key: string): void => void values.delete(key),
  },
});

const { isSeedPinned, setPinnedSeed } = await import('../src/sim/seed.ts');

after(() => {
  if (originalLocation) Object.defineProperty(globalThis, 'location', originalLocation);
  else Reflect.deleteProperty(globalThis, 'location');
  if (originalStorage) Object.defineProperty(globalThis, 'localStorage', originalStorage);
  else Reflect.deleteProperty(globalThis, 'localStorage');
});

test('current pin state follows pin and unpin after startup', () => {
  setPinnedSeed(null);
  assert.equal(isSeedPinned(17), false);
  setPinnedSeed(17);
  assert.equal(isSeedPinned(17), true);
  assert.equal(isSeedPinned(18), false);
  setPinnedSeed(null);
  assert.equal(isSeedPinned(17), false);
});

test('an authoritative URL seed outranks a stale local pin', () => {
  setPinnedSeed(17);
  (globalThis.location as { search: string }).search = '?seed=18';
  assert.equal(isSeedPinned(17), false);
  assert.equal(isSeedPinned(18), true);
});
