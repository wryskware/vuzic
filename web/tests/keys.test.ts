/**
 * Coverage for the pure keydown dispatch (handoffs/keybindings.md).
 *
 * `planKey` is the one place that knows which keys the app claims, so this
 * pins: the new `s` / `t` view toggles, the guarded transport claims already
 * in `main.ts`, the guard's superset over the old bare `INPUT` bail, and the
 * priority — the editable target beats every claim, then the explorer's keys
 * (live only while it is active) beat the shared `Space` / `H` / arrows.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';

import { isEditableTarget, planKey } from '../src/ui/keys.ts';
import type { KeyPlan, TargetInfo } from '../src/ui/keys.ts';

/** A `window` keydown: no `tagName`, not editable — the claims are wanted here. */
const BARE: TargetInfo = { tagName: undefined, isContentEditable: false };
const INPUT: TargetInfo = { tagName: 'INPUT', isContentEditable: false };
const TEXTAREA: TargetInfo = { tagName: 'TEXTAREA', isContentEditable: false };
const SELECT: TargetInfo = { tagName: 'SELECT', isContentEditable: false };
const EDITABLE: TargetInfo = { tagName: 'DIV', isContentEditable: true };
const BUTTON: TargetInfo = { tagName: 'BUTTON', isContentEditable: false };

function plan(
  code: string | null,
  target: TargetInfo = BARE,
  shiftKey = false,
  explorerActive = false,
) {
  return planKey(code, shiftKey, explorerActive, target);
}

test('s and t claim the view toggles from a bare target', () => {
  const s = plan('KeyS', BARE);
  assert.equal(s.kind, 'toggle-panel');
  assert.equal(s.preventDefault, true);
  const t = plan('KeyT', BARE);
  assert.equal(t.kind, 'toggle-timeline');
  assert.equal(t.preventDefault, true);
});

test('the existing transport claims are unchanged', () => {
  assert.deepEqual(plan('KeyH'), { kind: 'halt', preventDefault: true });
  assert.deepEqual(plan('Space'), { kind: 'play-toggle', preventDefault: true });
  assert.deepEqual(plan('ArrowLeft'), { kind: 'seek', delta: -2, preventDefault: true });
  assert.deepEqual(plan('ArrowRight'), { kind: 'seek', delta: 2, preventDefault: true });
  // Shift widens the seek, as the handler already did.
  assert.deepEqual(plan('ArrowLeft', BARE, true), { kind: 'seek', delta: -10, preventDefault: true });
  assert.deepEqual(plan('ArrowRight', BARE, true), { kind: 'seek', delta: 10, preventDefault: true });
});

test('every claimed key is preventDefaulted', () => {
  const claimed: KeyPlan[] = [
    plan('KeyS'),
    plan('KeyT'),
    plan('KeyH'),
    plan('Space'),
    plan('ArrowLeft'),
    plan('ArrowRight'),
    plan('KeyR', BARE, false, true),
    plan('Backspace', BARE, false, true),
    plan('Escape', BARE, false, true),
  ];
  for (const p of claimed) assert.equal(p.preventDefault, true);
});

test('unclaimed keys are left alone', () => {
  for (const code of ['KeyA', 'KeyE', 'F1', 'Enter', null]) {
    const p = plan(code);
    assert.equal(p.kind, 'none', `expected ${code ?? 'null'} to be unclaimed`);
    assert.equal(p.preventDefault, false);
  }
});

test('the guard is a superset of the old INPUT-only bail', () => {
  assert.equal(isEditableTarget(INPUT), true);
  assert.equal(isEditableTarget(TEXTAREA), true);
  assert.equal(isEditableTarget(SELECT), true);
  assert.equal(isEditableTarget(EDITABLE), true);
  // Not an editable form control, and not contenteditable — the claim stands.
  assert.equal(isEditableTarget(BARE), false);
  assert.equal(isEditableTarget(BUTTON), false);
  // `textarea` is the same element under any casing the DOM reports; the guard
  // keys on the element, not the spelling.
  assert.equal(isEditableTarget({ tagName: 'textarea', isContentEditable: false }), true);
});

test('typing s / t / h / space into a form control does not toggle anything', () => {
  for (const t of [INPUT, TEXTAREA, SELECT, EDITABLE]) {
    assert.equal(plan('KeyS', t).kind, 'none');
    assert.equal(plan('KeyT', t).kind, 'none');
    assert.equal(plan('KeyH', t).kind, 'none');
    assert.equal(plan('Space', t).kind, 'none');
    assert.equal(plan('ArrowLeft', t).kind, 'none');
  }
});

test('the explorer keys win while it is active, and nothing when it is not', () => {
  assert.equal(plan('KeyR', BARE, false, true).kind, 'explorer-reroll');
  assert.equal(plan('Backspace', BARE, false, true).kind, 'explorer-back');
  assert.equal(plan('Escape', BARE, false, true).kind, 'explorer-exit');

  assert.equal(plan('KeyR').kind, 'none');
  assert.equal(plan('Backspace').kind, 'none');
  assert.equal(plan('Escape').kind, 'none');

  // …and they are claimed while the explorer is up, even where `Space` would be.
  assert.equal(plan('Space', BARE, false, true).kind, 'play-toggle');
  assert.equal(plan('KeyH', BARE, false, true).kind, 'halt');
  // …and nothing else is claimed while it is active: the explorer is a
  // locked-in GUI, so the view toggles are meaningless over a full-screen grid.
  assert.equal(plan('KeyS', BARE, false, true).kind, 'none');
  assert.equal(plan('KeyT', BARE, false, true).kind, 'none');
  // …while they still toggle in plain mode, where the terrarium is on screen.
  assert.equal(plan('KeyS', BARE).kind, 'toggle-panel');
  assert.equal(plan('KeyT', BARE).kind, 'toggle-timeline');
});

test('the guard outranks the explorer too — typing never rerolls', () => {
  assert.equal(plan('KeyR', INPUT, false, true).kind, 'none');
  assert.equal(plan('Backspace', TEXTAREA, false, true).kind, 'none');
});
