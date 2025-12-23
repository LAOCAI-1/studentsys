import { Router } from "express";
import { verifyToken } from "../middlewares/auth.js";
import { success } from "../utils/response.js";

const router = Router();

// 先返回空列表，前端会自动回退到本地示例数据
router.get("/my", verifyToken, async (req, res) => {
  return success(res, [], "ok");
});

// 简单回显创建成功（无数据库持久化）
router.post("/", verifyToken, async (req, res) => {
  const body = req.body || {};
  const created = {
    id: Date.now(),
    type: body.repair_type || "其他",
    status: "已转派",
    time: new Date().toISOString().replace("T", " ").slice(0, 16),
    desc: body.description || "",
    location: `${body.zone || ""} ${body.building || ""} ${
      body.room || ""
    }`.trim(),
    repairer: "",
    completeTime: "",
    repairNote: "",
  };
  return success(res, created, "提交成功");
});

export default router;
