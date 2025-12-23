import jwt from "jsonwebtoken";
import { error } from "../utils/response.js";

const JWT_SECRET = process.env.JWT_SECRET || "dev-secret-change-me";

export function verifyToken(req, res, next) {
  try {
    const auth = req.headers["authorization"] || "";
    const token = auth.startsWith("Bearer ") ? auth.slice(7) : null;
    if (!token) return error(res, "缺少授权头", 401);

    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch (e) {
    return error(res, "无效或过期的令牌", 401);
  }
}
