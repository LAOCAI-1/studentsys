// backend/src/app.js
import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import apiRouter from "./routes/index.js"; // 汇总 /api 路由

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// 让后端从“项目根目录”的 .env 加载变量
const ROOT_DIR = path.join(__dirname, "..", "..");
dotenv.config({ path: path.join(ROOT_DIR, ".env") });

const app = express();

// CORS
app.use(
  cors({
    origin: "*",
    methods: ["GET", "POST", "PUT", "DELETE"],
    allowedHeaders: ["Content-Type", "Authorization"],
  })
);

// JSON 体积放大，容纳 base64 图片
app.use(express.json({ limit: "15mb" }));
app.use(express.urlencoded({ extended: true, limit: "15mb" }));

// 静态资源：指向 项目根/public
const PUBLIC_DIR = path.join(ROOT_DIR, "public");
console.log("[static] PUBLIC_DIR =", PUBLIC_DIR);
app.use(express.static(PUBLIC_DIR));

// API（/api/...）
app.use("/api", apiRouter);

// 主页跳转 dashboard
app.get("/", (_req, res) => res.redirect("/dashboard/"));

// 全局错误兜底（避免直接把栈回给前端）
app.use((err, _req, res, _next) => {
  console.error("[ERROR]", err);
  res.status(500).json({ ok: false, error: String(err?.message || err) });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`[server] booting...`);
  console.log(`✅ Server running at: http://localhost:${PORT}`);
  console.log(`👉 Login:              http://localhost:${PORT}/login/`);
  console.log(`👉 Dashboard:          http://localhost:${PORT}/dashboard/`);
  console.log(`👉 Profile:            http://localhost:${PORT}/profile/`);
  console.log(
    `👉 Face Health:        http://localhost:${PORT}/api/face/health`
  );
});
