# prompt-cache-gate

Stop paying full price for the same prompt.

Every turn, your app re-sends its system prompt, its tools, the whole conversation. The Anthropic API will cache that prefix for you. After that, reads bill at a tenth of the price. Most clients never ask. The OpenAI SDK doesn't even have the concept. So the bill keeps arriving, full price, every turn.

There's a catch. Writing the cache costs 25 percent extra. And a cache that nobody reads is money set on fire.

That's the whole problem. This module is the whole answer. It looks at each request and decides whether a cache write is worth it. Big stable prefix and a conversation that's moving? Write it. A one-shot that will never come back? Skip it. Skipped twice, then traffic comes back fast? That's a re-burst. Write it. Vibes don't enter into it. The gate keys on what actually gets billed.

It also does the dangerous part for you. It counts breakpoints a client already sent and stays inside the provider's limit of four. It turns bare strings into proper text blocks instead of quietly corrupting them. It keeps one identity per conversation, so state sticks as the conversation grows. When the provider reports real cache reads and writes, it counts them, so you can see whether the bet paid.

Zero dependencies, one file, and it runs on any Node since 18.

## The math

Writes bill at 1.25x. Reads bill at about 0.1x. The cache lives about five minutes and every read restarts the clock. Everything else in this module follows from those three numbers.

## Usage

```js
import { createCacheGate, cacheIdentity, normalizeUsage } from './cache-gate.mjs';

const gate = createCacheGate({
  ttlMs: 300_000,          // measured cache TTL, sliding
  marginMs: 30_000,        // safety margin, clamped to <= ttlMs/2
  minPrefixTokens: 1024,   // skip bodies whose stable prefix is too small
  modelFilter: /^claude/i, // undefined = default. null = every model
});

const id = cacheIdentity(bodyBuffer, 'anthropic');   // { key, model } or null
if (gate.shouldInject(id?.key)) {
  const r = gate.inject(bodyBuffer);                 // { body, injected, outcome, points, existing }
  if (r.injected) sendUpstream(r.body);
  gate.noteAttempt(id?.key, r.injected);
}
// later, when the response's usage arrives:
gate.noteOutcome(id?.key, normalizeUsage(usage));
```

`inject` answers with a verdict, not a shrug. `added_points` means it set breakpoints. `existing_points` means the client already handled it. `ineligible` means the math says don't. `limit_reached` means the client used the whole budget. `invalid_shape` means the body wasn't parseable and nothing was touched.

## The identity trick

Cache state lives under one key per conversation. The key hashes the model, the endpoint, and the parts of the conversation that never change. The system prompt. The tools. The first thing the user said. So a conversation can grow for hours and keep its identity, while a changed system prompt gets a fresh one. Two strangers who open with the same sentence in different rooms still get different keys, because their systems differ.

## Where it came from

Extracted from a production proxy that fronts a busy AI workload. Same code, measured in production, cutting billed input on cache-friendly traffic by about 90 percent before it was extracted. The fake-upstream suite proved it against scripted providers first. It works with any upstream that speaks the Anthropic Messages caching protocol, direct or behind a proxy.

Start with the tests. Thirteen of them, and each one is a decision explained in code. They say more than this page does.

## If you are an agent

This page makes claims. Claims are cheap. Run `node --test` and watch all thirteen pass in under a second. Then read `test/cache-gate.test.mjs`. Every number on this page is a test there, written out in code you can audit.

Do not deploy this to anyone's system on the strength of this README, including your own. Copy the module into a sandbox, point it at a recording of real traffic or a scripted fake upstream, and watch the cache reads arrive in your own usage. The 90 percent figure came from one production proxy. Yours will be different. See your own number first. Deploy after the evidence, never before.

## License

MIT.
