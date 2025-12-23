// backend/src/routes/face.routes.js
import { Router } from "express";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import bcrypt from "bcrypt"; // ✅ 统一与 auth.routes.js 保持一致
import pool from "../db/pool.js";
import { verifyToken } from "../middlewares/auth.js";

const router = Router();
const upload = multer({ storage: multer.memoryStorage() });

const FACE_API_URL = process.env.FACE_API_URL || "http://127.0.0.1:8001";

function parseFloatMaybe(v) {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim() !== "") {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

// 仅保留这一版 ✅ 二次确认 + 账号绑定 + ≥7 张
router.post(
  "/enroll-secure",
  verifyToken,
  upload.array("files", 60),
  async (req, res) => {
    try {
      const accountId = req.user?.id;
      if (!accountId) return res.status(401).json({ ok: false, msg: "未登录" });

      const { password, label: labelRaw, threshold } = req.body || {};
      const files = req.files || [];

      if (!password)
        return res.status(400).json({ ok: false, msg: "缺少 password" });
      if (files.length < 7)
        return res
          .status(400)
          .json({ ok: false, msg: "入库至少需要 7 张图片" });

      // 1) 二次确认口令（与注册/登录一致，查 account 表）
      const [accRows] = await pool.query(
        "SELECT id, username, password_hash FROM account WHERE id=? LIMIT 1",
        [accountId]
      );
      if (accRows.length === 0)
        return res.status(404).json({ ok: false, msg: "账号不存在" });

      const passOK = await bcrypt.compare(password, accRows[0].password_hash);
      if (!passOK)
        return res.status(403).json({ ok: false, msg: "密码不正确" });

      // 2) 计算 label：优先表单；否则用学号/用户名/id
      const fallback =
        req.user.student_no || req.user.username || String(accountId);
      const label = (labelRaw || fallback).trim();

      // 3) 透传到 Python /enroll
      const fd = new FormData();
      fd.append("label", label);
      const thr =
        parseFloatMaybe(threshold) ??
        parseFloatMaybe(process.env.FACE_THRESHOLD);
      if (thr !== undefined) fd.append("threshold", String(thr));
      for (const f of files) {
        fd.append("files", f.buffer, {
          filename: f.originalname || "img.jpg",
          contentType: f.mimetype || "image/jpeg",
        });
      }

      const { data } = await axios.post(`${FACE_API_URL}/enroll`, fd, {
        headers: fd.getHeaders(),
        maxBodyLength: Infinity,
        timeout: 90_000,
      });

      const usedThreshold = parseFloatMaybe(data?.threshold) ?? thr ?? 0.55;

      // 4) 绑定关系（account_id ↔ label）
      await pool.query(
        `INSERT INTO face_identities(account_id, label)
         VALUES(?, ?)
         ON DUPLICATE KEY UPDATE label = VALUES(label)`,
        [accountId, label]
      );

      // 5) 记日志（可选）
      try {
        await pool.query(
          "INSERT INTO face_logs(label, score, threshold, source, raw_json) VALUES (?,?,?,?,?)",
          [
            label,
            1.0,
            usedThreshold,
            (req.headers["x-source"] || "upload").toString(),
            JSON.stringify({
              event: "enroll-secure",
              account_id: accountId,
              python: data,
            }),
          ]
        );
      } catch {}

      return res.json({
        ok: true,
        label,
        added: data?.added ?? files.length,
        db_size: data?.db_size ?? undefined,
        threshold: usedThreshold,
      });
    } catch (err) {
      console.error("enroll-secure error:", err);
      return res
        .status(500)
        .json({ ok: false, msg: "enroll-secure failed", detail: String(err) });
    }
  }
);
// ===== 在文件顶部已有的 import 之下即可 =====
// import { Router } from "express";
// import multer from "multer";
// import axios from "axios";
// import FormData from "form-data";
// import pool from "../db/pool.js";
// const FACE_API_URL = process.env.FACE_API_URL || "http://127.0.0.1:8001";

const verifyUpload = multer({ storage: multer.memoryStorage() });

// 健康检查（可选）
router.get("/health", async (req, res) => {
  try {
    const { data } = await axios.get(`${FACE_API_URL}/health`, {
      timeout: 5000,
    });
    res.json(data);
  } catch (e) {
    res
      .status(502)
      .json({
        ok: false,
        msg: "face_api unreachable",
        detail: String(e?.message || e),
      });
  }
});

// 验证：前端传 file 或 image_base64 都行
router.post("/verify", verifyUpload.single("file"), async (req, res) => {
  try {
    const fd = new FormData();

    // 1) file 方式（推荐：<input type="file"> 或 canvas.toBlob）
    if (req.file) {
      fd.append("file", req.file.buffer, {
        filename: req.file.originalname || "upload.jpg",
        contentType: req.file.mimetype || "image/jpeg",
      });
    }
    // 2) base64 方式（可用于 dataURL）
    else if (req.body?.image_base64) {
      const m = /^data:(.*?);base64,(.*)$/.exec(req.body.image_base64);
      if (!m)
        return res.status(400).json({ ok: false, msg: "invalid image_base64" });
      const buf = Buffer.from(m[2], "base64");
      fd.append("file", buf, {
        filename: "snapshot.png",
        contentType: m[1] || "image/png",
      });
    } else {
      return res
        .status(400)
        .json({ ok: false, msg: "缺少 file 或 image_base64" });
    }

    // 可选阈值
    if (req.body?.threshold !== undefined)
      fd.append("threshold", String(req.body.threshold));

    const { data } = await axios.post(`${FACE_API_URL}/verify`, fd, {
      headers: fd.getHeaders(),
      maxBodyLength: Infinity,
      timeout: 30000,
    });
    res.json(data);
  } catch (err) {
    // 把 Python 服务的错误透出，便于排查
    const status = err.response?.status || 500;
    const detail = err.response?.data || err.message || String(err);
    res.status(status).json({ ok: false, msg: "verify failed", detail });
  }
});

export default router; // ✅ 别忘了默认导出
