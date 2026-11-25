// Client mẫu gọi Magician API có ký HMAC (Node >= 18, không cần dependency).
//   MAGICIAN_CLIENT_ID=dashboard MAGICIAN_SECRET=... node magician-client.mjs submit payload.json https://your-host/webhook
//   node magician-client.mjs status|wait|events <job_id>
//   node magician-client.mjs retry <job_id> [smart_ad_id=new_video_url ...]
//   node magician-client.mjs cancel <job_id> [smart_ad_id ...]
//   node magician-client.mjs list [state]
import { createHash, createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { readFileSync } from "node:fs";

export class MagicianClient {
  constructor(baseUrl, clientId, secret) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.clientId = clientId;
    this.secret = secret;
  }

  headers(method, pathQs, body) {
    const ts = Math.floor(Date.now() / 1000).toString();
    const nonce = randomBytes(16).toString("hex");
    const bodyHash = createHash("sha256").update(body).digest("hex");
    const msg = [ts, nonce, method.toUpperCase(), pathQs, bodyHash].join("\n");
    return {
      "X-Client-Id": this.clientId,
      "X-Timestamp": ts,
      "X-Nonce": nonce,
      "X-Signature": createHmac("sha256", this.secret).update(msg).digest("hex"),
    };
  }

  async request(method, path, params, jsonBody, extraHeaders) {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    const body = jsonBody === undefined ? Buffer.alloc(0) : Buffer.from(JSON.stringify(jsonBody));
    const headers = { ...this.headers(method, path + qs, body), ...(extraHeaders ?? {}) };
    if (body.length) headers["Content-Type"] = "application/json";
    const res = await fetch(this.baseUrl + path + qs, { method, headers, body: body.length ? body : undefined });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return res.json();
  }

  // idempotencyKey: gửi lại cùng key (vd retry khi mạng lỗi) thì nhận lại đúng job cũ
  submit(payload, callbackUrl, idempotencyKey) {
    return this.request("POST", "/jobs", callbackUrl ? { callback_url: callbackUrl } : undefined, payload,
      idempotencyKey ? { "Idempotency-Key": idempotencyKey } : undefined);
  }

  status(jobId) {
    return this.request("GET", `/jobs/${jobId}`);
  }

  list(state, limit = 50) {
    return this.request("GET", "/jobs", state ? { state, limit } : { limit });
  }

  events(jobId, after = 0) {
    return this.request("GET", `/jobs/${jobId}/events`, after ? { after } : undefined);
  }

  // videoUrls = { smart_ad_id: "link mới" } cho item lỗi source_expired
  retry(jobId, { smartAdIds, videoUrls = {}, force = false } = {}) {
    const body = { video_urls: videoUrls, force };
    if (smartAdIds) body.smart_ad_ids = smartAdIds;
    return this.request("POST", `/jobs/${jobId}/retry`, undefined, body);
  }

  // Huỷ / kill: bỏ trống smartAdIds = huỷ cả job
  cancel(jobId, smartAdIds) {
    return this.request("POST", `/jobs/${jobId}/cancel`, undefined, smartAdIds ? { smart_ad_ids: smartAdIds } : undefined);
  }

  async wait(jobId, { timeoutMs = 1_800_000, intervalMs = 5000 } = {}) {
    const end = Date.now() + timeoutMs;
    for (;;) {
      const job = await this.status(jobId);
      if (["done", "cancelled", "failed"].includes(job.state) || Date.now() > end) return job;
      await new Promise((r) => setTimeout(r, intervalMs));
    }
  }
}

// Kiểm tra webhook: dùng raw body (Buffer), KHÔNG dùng JSON đã parse rồi stringify lại.
export function verifyWebhook(secret, rawBody, timestamp, signature, maxAge = 300) {
  if (!/^\d+$/.test(timestamp ?? "") || Math.abs(Date.now() / 1000 - Number(timestamp)) > maxAge) return false;
  const expected = "sha256=" + createHmac("sha256", secret).update(`${timestamp}.`).update(rawBody).digest("hex");
  const a = Buffer.from(expected), b = Buffer.from(signature ?? "");
  return a.length === b.length && timingSafeEqual(a, b);
}

if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith("magician-client.mjs")) {
  const c = new MagicianClient(process.env.MAGICIAN_URL ?? "https://magician.qhuy204.id.vn",
    process.env.MAGICIAN_CLIENT_ID, process.env.MAGICIAN_SECRET);
  const [cmd, arg, ...rest] = process.argv.slice(2);
  const urls = Object.fromEntries(rest.filter((a) => a.includes("=")).map((a) => [a.slice(0, a.indexOf("=")), a.slice(a.indexOf("=") + 1)]));
  const run = {
    submit: () => c.submit(JSON.parse(readFileSync(arg, "utf8")), rest[0], rest[1]),
    status: () => c.status(arg),
    wait: () => c.wait(arg),
    events: () => c.events(arg),
    retry: () => c.retry(arg, Object.keys(urls).length ? { smartAdIds: Object.keys(urls), videoUrls: urls } : {}),
    cancel: () => c.cancel(arg, rest.length ? rest : undefined),
    list: () => c.list(arg),
  }[cmd];
  if (!run) throw new Error(`unknown command ${cmd}`);
  console.log(JSON.stringify(await run(), null, 2));
}
