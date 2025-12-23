// backend/src/routes/takeout.routes.js
import { Router } from "express";
import pool from "../db/pool.js";

const router = Router();

const INTERNAL_API_KEY = process.env.INTERNAL_API_KEY || "";

// 给 Python 脚本调用：简单内部鉴权
function requireInternalKey(req, res, next) {
  if (!INTERNAL_API_KEY) return next(); // 不设置 key 时放行（开发方便）；上线建议必须设置
  const key = req.headers["x-internal-key"];
  if (key !== INTERNAL_API_KEY) {
    return res.status(401).json({ ok: false, msg: "invalid internal key" });
  }
  next();
}

/**
 * Python -> Node：写入一条取件事件
 * POST /api/internal/takeout/event
 * body:
 *  {
 *    account_id?: number,
 *    label?: string,
 *    similarity?: number,
 *    status?: "pickup"|"not_pickup"|"unknown",
 *    source?: "camera"|"upload",
 *    snapshot_url?: string,
 *    raw_json?: object
 *  }
 */
router.post("/internal/takeout/event", requireInternalKey, async (req, res) => {
  try {
    const {
      account_id: accountIdRaw,
      label = null,
      similarity = null,
      status = "unknown",
      source = "camera",
      snapshot_url = null,
      raw_json = null,
    } = req.body || {};

    // status/source 校验（避免 enum 报错）
    const statusOk = ["pickup", "not_pickup", "unknown"].includes(status);
    const sourceOk = ["camera", "upload"].includes(source);
    if (!statusOk)
      return res.status(400).json({ ok: false, msg: "bad status" });
    if (!sourceOk)
      return res.status(400).json({ ok: false, msg: "bad source" });

    let accountId = accountIdRaw ?? null;

    // 可选：如果没传 account_id，尝试用 label 去反查 face_identities（如果你建了这个表）
    if (accountId == null && label) {
      try {
        const [rows] = await pool.query(
          "SELECT account_id FROM face_identities WHERE label=? LIMIT 1",
          [label]
        );
        if (rows?.length) accountId = rows[0].account_id;
      } catch (_) {
        // 没建 face_identities 或查询失败：忽略即可
      }
    }

    const rawJson = raw_json == null ? null : JSON.stringify(raw_json);

    await pool.query(
      `INSERT INTO takeout_events
        (account_id, label, similarity, status, source, snapshot_url, raw_json)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [accountId, label, similarity, status, source, snapshot_url, rawJson]
    );

    res.json({ ok: true });
  } catch (err) {
    console.error("[takeout/event] error:", err);
    res.status(500).json({ ok: false, msg: "server error" });
  }
});

/**
 * 前端：取最新事件
 * GET /api/takeout/events?limit=50
 */
router.get("/takeout/events", async (req, res) => {
  try {
    const limit = Math.min(parseInt(req.query.limit || "50", 10), 200);
    const [rows] = await pool.query(
      `SELECT id, account_id, label, similarity, status, source, snapshot_url, raw_json, created_at
       FROM takeout_events
       ORDER BY id DESC
       LIMIT ?`,
      [limit]
    );
    res.json({ ok: true, rows });
  } catch (err) {
    console.error("[takeout/events] error:", err);
    res.status(500).json({ ok: false, msg: "server error" });
  }
});

/**
 * 前端：统计（最近 minutes 分钟）
 * GET /api/takeout/stats?minutes=60
 */
router.get("/takeout/stats", async (req, res) => {
  try {
    const minutes = Math.min(parseInt(req.query.minutes || "60", 10), 24 * 60);

    const [byStatus] = await pool.query(
      `SELECT status, COUNT(*) AS cnt
       FROM takeout_events
       WHERE created_at >= NOW() - INTERVAL ? MINUTE
       GROUP BY status`,
      [minutes]
    );

    const [topLabels] = await pool.query(
      `SELECT label, COUNT(*) AS cnt
       FROM takeout_events
       WHERE created_at >= NOW() - INTERVAL ? MINUTE
       GROUP BY label
       ORDER BY cnt DESC
       LIMIT 20`,
      [minutes]
    );

    res.json({ ok: true, minutes, byStatus, topLabels });
  } catch (err) {
    console.error("[takeout/stats] error:", err);
    res.status(500).json({ ok: false, msg: "server error" });
  }
});

export default router;
