import test from 'node:test';
import assert from 'node:assert/strict';
import { createCacheGate, cacheIdentity, normalizeUsage, ANTHROPIC_BREAKPOINT_LIMIT } from '../cache-gate.mjs';

const SYSTEM = 'You are a helpful assistant. '.repeat(200); // ~5600 chars
const tool = { name: 'get_weather', description: 'x'.repeat(400), input_schema: { type: 'object' } };

function body(overrides = {}) {
  return {
    model: 'claude-sonnet-5',
    max_tokens: 16,
    system: SYSTEM,
    tools: [tool],
    messages: [
      { role: 'user', content: 'say A' },
      { role: 'assistant', content: 'A' },
      { role: 'user', content: 'say B' },
    ],
    ...overrides,
  };
}

test('normalizeUsage: anthropic shape with cost-weighted input', () => {
  const n = normalizeUsage({ input_tokens: 100, output_tokens: 10, cache_read_input_tokens: 1000, cache_creation_input_tokens: 40 });
  assert.equal(n.promptTotal, 1140);
  assert.equal(n.total, 1150);
  assert.equal(n.billedInput, 100 + 100 + 40 * 1.25);
});

test('normalizeUsage: openai shape falls back to prompt_tokens_details', () => {
  const n = normalizeUsage({ prompt_tokens: 500, completion_tokens: 20, prompt_tokens_details: { cached_tokens: 300 } });
  assert.equal(n.input, 200);
  assert.equal(n.cacheRead, 300);
  assert.equal(n.output, 20);
  assert.equal(n.billedInput, null);
});

test('inject: ineligible one-message body is untouched', () => {
  const gate = createCacheGate();
  const b = body({ messages: [{ role: 'user', content: 'hi' }] });
  const r = gate.inject(JSON.stringify(b));
  assert.equal(r.injected, false);
  assert.equal(r.outcome, 'ineligible');
  assert.equal(r.body, JSON.stringify(b)); // same string reference: untouched
});

test('inject: prefix below threshold is ineligible', () => {
  const gate = createCacheGate({ minPrefixTokens: 100000 });
  const r = gate.inject(JSON.stringify(body()));
  assert.equal(r.outcome, 'ineligible');
  assert.ok(r.prefixTokens < 100000);
});

test('inject: non-matching model passes through untouched', () => {
  const gate = createCacheGate();
  const b = body({ model: 'gpt-x' });
  const r = gate.inject(JSON.stringify(b));
  assert.equal(r.injected, false);
  assert.equal(r.outcome, 'ineligible');
});

test('inject: eligible body gets tool + system + final-message breakpoints', () => {
  const gate = createCacheGate();
  const r = gate.inject(JSON.stringify(body()));
  assert.equal(r.injected, true);
  assert.equal(r.outcome, 'added_points');
  assert.equal(r.points, 3);
  const o = JSON.parse(r.body.toString('utf8'));
  assert.equal(o.tools[0].cache_control.type, 'ephemeral');
  assert.equal(o.system[0].type, 'text');
  assert.equal(o.system[0].cache_control.type, 'ephemeral');
  const last = o.messages[o.messages.length - 1];
  assert.equal(last.content[0].cache_control.type, 'ephemeral');
  // original buffer untouched
  const orig = JSON.parse(JSON.stringify(body()));
  assert.ok(!JSON.stringify(orig).includes('ephemeral'));
});

test('inject: string content entries rebuilt as text blocks, never spread', () => {
  const gate = createCacheGate();
  const r = gate.inject(JSON.stringify(body()));
  const o = JSON.parse(r.body.toString('utf8'));
  const last = o.messages[o.messages.length - 1];
  assert.equal(last.content[0].text, 'say B');
  assert.equal(typeof last.content[0].cache_control, 'object');
});

test('inject: client breakpoints preserved and provider limit respected', () => {
  const gate = createCacheGate();
  const b = body({
    tools: [
      { ...tool, cache_control: { type: 'ephemeral' } },
      { ...tool, cache_control: { type: 'ephemeral' } },
      { ...tool, cache_control: { type: 'ephemeral' } },
      tool,
    ],
  });
  const r = gate.inject(JSON.stringify(b));
  assert.equal(r.injected, true);
  assert.equal(r.existing, 3);
  assert.equal(r.points, ANTHROPIC_BREAKPOINT_LIMIT - 3); // adds only within budget
  const o = JSON.parse(r.body.toString('utf8'));
  assert.equal(o.tools[0].cache_control.type, 'ephemeral');
});

test('inject: limit_reached when client already used the budget', () => {
  const gate = createCacheGate();
  const cc = { type: 'ephemeral' };
  const b = body({
    tools: [
      { ...tool, cache_control: cc }, { ...tool, cache_control: cc },
      { ...tool, cache_control: cc }, { ...tool, cache_control: cc },
    ],
  });
  const r = gate.inject(JSON.stringify(b));
  assert.equal(r.injected, false);
  assert.equal(r.outcome, 'limit_reached');
  assert.equal(r.existing, 4);
});

test('gate: fresh injects, inside TTL injects, expired isolated skips, re-burst injects', () => {
  const gate = createCacheGate({ ttlMs: 300000, marginMs: 30000 });
  const id = cacheIdentity(JSON.stringify(body()), 'anthropic');
  const t0 = 1_000_000;
  assert.equal(gate.shouldInject(id.key, t0), true); // fresh
  gate.noteAttempt(id.key, true, t0);
  assert.equal(gate.shouldInject(id.key, t0 + 60_000), true); // inside margin
  assert.equal(gate.shouldInject(id.key, t0 + 600_000), false); // expired, isolated
  // re-burst: a skipped request followed by a quick return injects again
  gate.noteAttempt(id.key, false, t0 + 600_000);
  assert.equal(gate.shouldInject(id.key, t0 + 620_000), true);
});

test('gate: attempts and confirmations are separate counters', () => {
  const gate = createCacheGate();
  const id = cacheIdentity(JSON.stringify(body()), 'anthropic');
  gate.inject(JSON.stringify(body()));
  gate.noteAttempt(id.key, true);
  gate.noteOutcome(id.key, normalizeUsage({ input_tokens: 10, output_tokens: 1, cache_read_input_tokens: 1775 }));
  const s = gate.stats();
  assert.equal(s.injected, 1);
  assert.equal(s.cacheRead, 1);
  assert.equal(s.cacheReadTokens, 1775);
  assert.equal(s.cacheCreated, 0);
});

test('identity: growing conversation keeps one key; model or system change does not', () => {
  const b1 = body();
  const b2 = body({
    messages: [...b1.messages, { role: 'assistant', content: 'B' }, { role: 'user', content: 'say C' }],
  });
  const k1 = cacheIdentity(JSON.stringify(b1), 'anthropic');
  const k2 = cacheIdentity(JSON.stringify(b2), 'anthropic');
  assert.equal(k1.key, k2.key); // prefix is identical, only the trailing turn grew

  const kModel = cacheIdentity(JSON.stringify(body({ model: 'claude-haiku-4' })), 'anthropic');
  assert.notEqual(k1.key, kModel.key);

  const kSys = cacheIdentity(JSON.stringify(body({ system: SYSTEM + 'extra' })), 'anthropic');
  assert.notEqual(k1.key, kSys.key);
});

test('identity: null for bodies with nothing usable', () => {
  assert.equal(cacheIdentity(JSON.stringify({ model: 'claude-sonnet-5', messages: [] })), null);
  assert.equal(cacheIdentity('not json'), null);
});
