// Client mẫu gọi Magician API có ký HMAC (Node >= 18, không cần dependency).
//   MAGICIAN_CLIENT_ID=dashboard MAGICIAN_SECRET=... node magician-client.mjs submit payload.json https://your-host/webhook
//   node magician-client.mjs status <job_id>
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

  async request(method, path, params, jsonBody) {
    const qs = params ? "?" + new URLSearchParams(params).toString() : "";
    const body = jsonBody === undefined ? Buffer.alloc(0) : Buffer.from(JSON.stringify(jsonBody));
    const headers = this.headers(method, path + qs, body);
    if (body.length) headers["Content-Type"] = "application/json";
    const res = await fetch(this.baseUrl + path + qs, { method, headers, body: body.length ? body : undefined });
    if (!res.ok) throw new Error(`${res.status} ${await res.text()}`);
    return res.json();
  }

  submit(payload, callbackUrl) {
    return this.request("POST", "/jobs", callbackUrl ? { callback_url: callbackUrl } : undefined, payload);
  }

  status(jobId) {
    return this.request("GET", `/jobs/${jobId}`);
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
  const [cmd, arg, cb] = process.argv.slice(2);
  const out = cmd === "submit" ? await c.submit(JSON.parse(readFileSync(arg, "utf8")), cb) : await c.status(arg);
  console.log(JSON.stringify(out, null, 2));
}
