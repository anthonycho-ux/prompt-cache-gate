// prompt-cache-gate — reuse-gated Anthropic prompt-cache injection.
//
// Extracted from a production multi-account proxy (2026-09). The upstream
// provider (Anthropic Messages API) supports explicit prompt caching:
// clients mark stable prefixes with `cache_control: {type: "ephemeral"}`
// breakpoints. Writes bill at a 1.25x premium, reads at ~0.1x, and the
// cache lives ~5 minutes sliding. Clients that never set breakpoints
// (most OpenAI-SDK-based tools) pay full input price on every turn.
//
// This module decides WHEN writing that cache is worth it, and mutates
// request bodies to set breakpoints safely:
//   - identity:  model + endpoint mode + stable-prefix fingerprint
//   - gate:      inject on fresh conversations and bursts, skip isolated
//                requests that arrive after the TTL (a one-shot never
//                reads its cache; an unread write is pure premium)
//   - safety:    client-sent breakpoints are counted and preserved;
//                injection adds only within the provider limit (4);
//                string entries are rebuilt as text blocks, never spread
//   - evidence:  attempts and confirmed cache writes/reads are separate
//                counters so hit rate is measurable
//
// Zero dependencies. Node 18+.

import crypto from 'node:crypto';

export const ANTHROPIC_BREAKPOINT_LIMIT = 4;

// Rough chars-per-token for prefix estimation. Deliberately a heuristic:
// the threshold is configurable, so this is a gate, not a tokenizer.
const CHARS_PER_TOKEN = 4;
const AFFINITY_PREFIX_CHARS = 2048;

/**
 * ONE normalized usage shape for every response path.
 *   input       uncached input tokens
 *   cacheRead   tokens served from the prompt cache
 *   cacheWrite  tokens written to the prompt cache
 *   output      completion/output tokens
 *   promptTotal logical prompt = input + cacheRead + cacheWrite
 *   billedInput cost-weighted input (reads ~0.1x, writes 1.25x);
 *               null when pricing is unknown (OpenAI-shaped usage)
 *   total       promptTotal + output
 */
export function normalizeUsage(u) {
  if (!u || typeof u !== 'object') return null;
  const isAnthropic = Number.isFinite(u.input_tokens);
  const cacheRead = Number.isFinite(u.cache_read_input_tokens)
    ? u.cache_read_input_tokens
    : Number.isFinite(u.prompt_tokens_details?.cached_tokens)
      ? u.prompt_tokens_details.cached_tokens
      : 0;
  const cacheWrite = Number.isFinite(u.cache_creation_input_tokens)
    ? u.cache_creation_input_tokens
    : 0;
  let input, output;
  if (isAnthropic) {
    input = u.input_tokens || 0;
    output = Number.isFinite(u.output_tokens) ? u.output_tokens : 0;
  } else {
    const prompt = Number.isFinite(u.prompt_tokens) ? u.prompt_tokens : 0;
    input = Math.max(0, prompt - cacheRead);
    output = Number.isFinite(u.completion_tokens) ? u.completion_tokens : 0;
  }
  const promptTotal = input + cacheRead + cacheWrite;
  return {
    input,
    cacheRead,
    cacheWrite,
    output,
    promptTotal,
    billedInput: isAnthropic ? input + cacheRead * 0.1 + cacheWrite * 1.25 : null,
    total: promptTotal + output,
  };
}

/**
 * Build a cache identity from an Anthropic /v1/messages (or close-enough
 * OpenAI) body: hash of model + endpoint mode + a growth-stable fingerprint
 * (system, tools, and the FIRST user message — the parts that stay constant
 * while a conversation appends turns). A growing conversation keeps one
 * identity; a system/tool/model change gets a fresh one. Returns
 * { key, model } or null when nothing usable exists.
 */
export function cacheIdentity(body, endpointMode = 'anthropic') {
  try {
    const o = typeof body === 'string' ? JSON.parse(body) : JSON.parse(body.toString('utf8'));
    if (!o || typeof o !== 'object') return null;
    const fp = stablePrefixFingerprint(o);
    if (!fp) return null;
    const key = crypto
      .createHash('sha1')
      .update(`${String(o.model || '')}|${endpointMode}|${fp}`)
      .digest('hex');
    return { key, model: String(o.model || '') };
  } catch {
    return null;
  }
}

function stablePrefixFingerprint(o) {
  const parts = [];
  if (o.system) parts.push(typeof o.system === 'string' ? o.system : JSON.stringify(o.system));
  if (Array.isArray(o.tools) && o.tools.length) parts.push(JSON.stringify(o.tools));
  if (Array.isArray(o.messages)) {
    const firstUser = o.messages.find((m) => m && m.role === 'user');
    if (firstUser) {
      const content =
        typeof firstUser.content === 'string' ? firstUser.content : JSON.stringify(firstUser.content);
      if (content) parts.push(String(content).slice(0, AFFINITY_PREFIX_CHARS));
    }
  }
  if (!parts.length) return null;
  return crypto.createHash('sha1').update(parts.join('\u0000')).digest('hex');
}

/**
 * The gate. Owns per-identity injection state (bounded, true LRU with
 * expiry-aware eviction) and the decision state machine:
 *   fresh conversation            -> inject (first write, betting on a burst)
 *   last injection inside the TTL -> inject (read + refresh)
 *   skipped recently but back fast (re-burst) -> inject
 *   otherwise                     -> skip
 * Skipping keys on lastInjectedAt, not last request: a conversation whose
 * every turn lands after the TTL stays skipped instead of oscillating
 * skip/cold-write (+25% for nothing).
 */
export function createCacheGate(options = {}) {
  const ttlMs = options.ttlMs ?? 5 * 60 * 1000;
  const marginMs = Math.min(
    Math.max(options.marginMs ?? 30 * 1000, 0),
    Math.floor(ttlMs / 2)
  );
  const minPrefixTokens = options.minPrefixTokens ?? 1024;
  const minPrefixTokensByClass = options.minPrefixTokensByClass ?? {};
  // undefined -> default Anthropic filter; null -> match every model.
  const modelFilter = options.modelFilter === undefined ? /^claude/i : options.modelFilter;
  const breakpointLimit = options.breakpointLimit ?? ANTHROPIC_BREAKPOINT_LIMIT;
  const maxKeys = options.maxKeys ?? 500;
  const keyTtlMs = options.keyTtlMs ?? 30 * 60 * 1000;
  const charsPerToken = options.charsPerToken ?? CHARS_PER_TOKEN;

  const injectState = new Map(); // key -> { lastInjectedAt, lastSkippedAt, lastRequestAt }
  const counters = {
    injected: 0, breakpointAdded: 0, requestSent: 0, requestFailed: 0,
    cacheCreated: 0, cacheRead: 0, cacheReadTokens: 0, cacheWriteTokens: 0,
  };

  const freshState = () => ({ lastInjectedAt: null, lastSkippedAt: null, lastRequestAt: null });

  function purgeExpiredLRU(map, max) {
    const now = Date.now();
    for (const [k, v] of map) {
      if (map.size <= max) break;
      const at = v.lastRequestAt;
      if (Number.isFinite(at) && now - at > keyTtlMs) map.delete(k);
    }
    while (map.size > max) map.delete(map.keys().next().value);
  }

  /** Decision only. See the state machine comment above. */
  function shouldInject(key, now = Date.now()) {
    if (!key) return true; // no identity: conservative — inject
    const st = injectState.get(key) || null;
    const margin = ttlMs - marginMs;
    if (!st || st.lastInjectedAt === null) return true;
    if (st.lastInjectedAt !== null && now - st.lastInjectedAt < margin) return true;
    if (
      st.lastSkippedAt !== null &&
      st.lastRequestAt !== null &&
      now - st.lastRequestAt < margin &&
      now - st.lastSkippedAt < margin
    ) {
      return true;
    }
    return false;
  }

  /** Record the decision for this request (an ATTEMPT, not an outcome). */
  function noteAttempt(key, injected, now = Date.now()) {
    if (!key) return;
    const st = injectState.get(key) || freshState();
    st.lastRequestAt = now;
    if (injected) st.lastInjectedAt = now;
    else st.lastSkippedAt = now;
    const hit = injectState.get(key);
    if (hit) injectState.delete(key); // LRU refresh
    else if (injectState.size >= maxKeys) purgeExpiredLRU(injectState, maxKeys);
    injectState.set(key, st);
  }

  /** Fold confirmed upstream evidence (normalized usage) into counters. */
  function noteOutcome(key, usage) {
    if (!usage) return;
    if (usage.cacheWrite > 0) {
      counters.cacheCreated += 1;
      counters.cacheWriteTokens += usage.cacheWrite;
    } else if (usage.cacheRead > 0) {
      counters.cacheRead += 1;
      counters.cacheReadTokens += usage.cacheRead;
    }
    if (usage.cacheWrite > 0 || usage.cacheRead > 0) {
      const st = injectState.get(key);
      if (st) {
        injectState.delete(key);
        injectState.set(key, st); // keep live identities warm in LRU order
      }
    }
  }

  function minPrefixTokensFor(model) {
    const cls = String(model || '').split(/[-_.]/)[0].toLowerCase();
    return minPrefixTokensByClass[cls] ?? minPrefixTokens;
  }

  function estimateStablePrefixTokens(o) {
    let chars = 0;
    if (Array.isArray(o.tools) && o.tools.length) chars += JSON.stringify(o.tools).length;
    if (o.system) chars += typeof o.system === 'string' ? o.system.length : JSON.stringify(o.system).length;
    if (Array.isArray(o.messages)) {
      let lastUser = -1;
      for (let i = o.messages.length - 1; i >= 0; i--) {
        if (o.messages[i] && o.messages[i].role === 'user') { lastUser = i; break; }
      }
      if (lastUser > 0) chars += JSON.stringify(o.messages.slice(0, lastUser)).length;
    }
    return Math.ceil(chars / charsPerToken);
  }

  function countCacheControlPoints(o) {
    let n = 0;
    const walk = (v) => {
      if (!v || typeof v !== 'object') return;
      if (Array.isArray(v)) {
        for (const item of v) walk(item);
        return;
      }
      if (v.cache_control) n += 1;
      for (const k of Object.keys(v)) {
        if (k === 'cache_control') continue;
        walk(v[k]);
      }
    };
    walk(o);
    return n;
  }

  const CC = { type: 'ephemeral' };

  /**
   * Inject cache_control breakpoints: last tool, last system block, final
   * message block. Never mutates the input buffer. Outcomes:
   *   added_points     breakpoints were added
   *   existing_points  eligible but every target already had one
   *   ineligible       wrong model / <2 messages / prefix below threshold
   *   invalid_shape    unparseable JSON
   *   limit_reached    client already used the breakpoint budget
   */
  function inject(bodyBuf) {
    let o;
    try {
      o = typeof bodyBuf === 'string' ? JSON.parse(bodyBuf) : JSON.parse(bodyBuf.toString('utf8'));
    } catch {
      return { body: bodyBuf, injected: false, outcome: 'invalid_shape', points: 0, existing: 0 };
    }
    if (!o || typeof o !== 'object' || (modelFilter && !modelFilter.test(String(o.model || '')))) {
      return { body: bodyBuf, injected: false, outcome: 'ineligible', points: 0, existing: 0, model: o?.model };
    }
    if (!Array.isArray(o.messages) || o.messages.length < 2 || !o.system) {
      return { body: bodyBuf, injected: false, outcome: 'ineligible', points: 0, existing: 0, model: o.model };
    }
    const minTokens = minPrefixTokensFor(o.model);
    const prefixTokens = estimateStablePrefixTokens(o);
    if (prefixTokens < minTokens) {
      return {
        body: bodyBuf, injected: false, outcome: 'ineligible',
        points: 0, existing: 0, model: o.model, prefixTokens, minTokens,
      };
    }
    const existing = countCacheControlPoints(o);
    if (existing >= breakpointLimit) {
      return { body: bodyBuf, injected: false, outcome: 'limit_reached', points: 0, existing, model: o.model };
    }
    const budget = breakpointLimit - existing;
    let n = 0;
    const room = () => n < budget;
    if (Array.isArray(o.tools) && o.tools.length && room()) {
      const last = o.tools.length - 1;
      if (o.tools[last] && !o.tools[last].cache_control) {
        o.tools[last] = { ...o.tools[last], cache_control: CC };
        n++;
      }
    }
    if (room()) {
      if (typeof o.system === 'string' && o.system) {
        o.system = [{ type: 'text', text: o.system, cache_control: CC }];
        n++;
      } else if (Array.isArray(o.system) && o.system.length) {
        const last = o.system.length - 1;
        const s = o.system[last];
        if (s && typeof s === 'string' && s) {
          o.system = [...o.system.slice(0, last), { type: 'text', text: s, cache_control: CC }];
          n++;
        } else if (s && typeof s === 'object' && !s.cache_control) {
          o.system = [...o.system.slice(0, last), { ...s, cache_control: CC }];
          n++;
        }
      }
    }
    if (room() && Array.isArray(o.messages) && o.messages.length) {
      const mi = o.messages.length - 1;
      const m = o.messages[mi];
      if (m && Array.isArray(m.content) && m.content.length) {
        const ci = m.content.length - 1;
        const c = m.content[ci];
        if (c && typeof c === 'string' && c) {
          m.content = [...m.content.slice(0, ci), { type: 'text', text: c, cache_control: CC }];
          n++;
        } else if (c && typeof c === 'object' && !c.cache_control) {
          m.content = [...m.content.slice(0, ci), { ...c, cache_control: CC }];
          n++;
        }
      } else if (m && typeof m.content === 'string' && m.content) {
        m.content = [{ type: 'text', text: m.content, cache_control: CC }];
        n++;
      }
    }
    if (!n) {
      return {
        body: bodyBuf, injected: false,
        outcome: existing ? 'existing_points' : 'ineligible',
        points: 0, existing, model: o.model,
      };
    }
    if (modelFilter && modelFilter.test(String(o.model || ''))) counters.injected += 1;
    counters.breakpointAdded += n;
    return {
      body: Buffer.from(JSON.stringify(o), 'utf8'),
      injected: true,
      outcome: 'added_points',
      model: o.model,
      points: n,
      existing,
    };
  }

  function stats() {
    return { ...counters, keys: injectState.size, ttlMs, marginMs, minPrefixTokens, breakpointLimit };
  }

  return { cacheIdentity, shouldInject, noteAttempt, noteOutcome, inject, stats };
}
