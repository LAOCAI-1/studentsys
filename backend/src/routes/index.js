// backend/src/routes/index.js
import { Router } from "express";
import authRoutes from "./auth.routes.js";
import faceRoutes from "./face.routes.js";
import takeoutRoutes from "./takeout.routes.js";

const router = Router();

// /api/auth/*
router.use("/auth", authRoutes);

// /api/face/*   （face.routes.js 里定义 /health /verify /enroll /logs）
router.use("/face", faceRoutes);
// ✅ 这里用 "/"，因为 takeout.routes.js 内部已经写了 /takeout/* 和 /internal/takeout/*
router.use("/", takeoutRoutes);

export default router;
