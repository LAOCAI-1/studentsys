import express from "express";
import jwt from "jsonwebtoken";
import bcrypt from "bcrypt";
import pool from "../db/pool.js";
import { success, error } from "../utils/response.js";
import { verifyToken } from "../middlewares/auth.js";

const router = express.Router();
const JWT_SECRET = process.env.JWT_SECRET || "dev-secret-change-me";

// ============ 注册 ============
// body: { student_no, real_name, college, major, grade, building_no, room_no, password, phone }
router.post("/register", async (req, res) => {
  try {
    const {
      student_no,
      real_name,
      college,
      major,
      grade,
      building_no,
      room_no,
      password,
      phone,
    } = req.body;

    if (
      !student_no ||
      !real_name ||
      !major ||
      !grade ||
      !building_no ||
      !room_no ||
      !password
    ) {
      return error(res, "缺少必填字段", 400);
    }

    // 学号唯一性
    const [exists] = await pool.query(
      "SELECT id FROM student_profile WHERE student_no=?",
      [student_no]
    );
    if (exists.length > 0) return error(res, "该学号已注册", 400);

    // 账号：account
    const hash = await bcrypt.hash(password, 10);
    const [accRes] = await pool.query(
      "INSERT INTO account (username, password_hash, phone, role) VALUES (?, ?, ?, 'student')",
      [student_no, hash, phone]
    );

    // 宿舍：dorm_room（查/建）
    const [dormRows] = await pool.query(
      "SELECT id FROM dorm_room WHERE building_no=? AND room_no=?",
      [building_no, room_no]
    );
    let dorm_id = dormRows?.[0]?.id;
    if (!dorm_id) {
      const [insDorm] = await pool.query(
        "INSERT INTO dorm_room (building_no, room_no) VALUES (?, ?)",
        [building_no, room_no]
      );
      dorm_id = insDorm.insertId;
    }

    // 学生档案：student_profile
    await pool.query(
      `INSERT INTO student_profile
       (account_id, real_name, student_no, college, major, grade, dorm_room_id)
       VALUES (?,?,?,?,?,?,?)`,
      [accRes.insertId, real_name, student_no, college, major, grade, dorm_id]
    );

    success(res, null, "注册成功");
  } catch (err) {
    console.error("注册失败:", err);
    error(res, "服务器错误", 500);
  }
});

// ============ 登录 ============
// 兼容两种 body：
// 学生：{ student_no, password }
// 管理员：{ username, password }
router.post("/login", async (req, res) => {
  const { student_no, username, password } = req.body || {};

  try {
    if (!(student_no || username) || !password) {
      return error(res, "账号与密码不能为空", 400);
    }

    // 如果有 student_no => 学生登录
    if (student_no) {
      const [rows] = await pool.query(
        `SELECT a.id, a.password_hash, a.role,
                s.real_name, s.college, s.major, s.grade, s.student_no,
                d.building_no, d.room_no
         FROM account a
         JOIN student_profile s ON a.id = s.account_id
         LEFT JOIN dorm_room d ON s.dorm_room_id = d.id
         WHERE s.student_no=? AND a.role='student'`,
        [student_no]
      );
      if (rows.length === 0) return error(res, "账号不存在", 400);

      const u = rows[0];
      const ok = await bcrypt.compare(password, u.password_hash);
      if (!ok) return error(res, "密码错误", 400);

      const token = jwt.sign(
        {
          id: u.id,
          student_no: u.student_no,
          real_name: u.real_name,
          role: "student",
        },
        JWT_SECRET,
        { expiresIn: "7d" }
      );

      return success(
        res,
        {
          token,
          user: {
            real_name: u.real_name,
            college: u.college,
            major: u.major,
            grade: u.grade,
            student_no: u.student_no,
            building_no: u.building_no,
            room_no: u.room_no,
          },
        },
        "登录成功"
      );
    }

    // 否则按 username 处理（用于管理员登录）
    const [accRows] = await pool.query(
      "SELECT id, username, password_hash, role FROM account WHERE username=?",
      [username]
    );
    if (accRows.length === 0) return error(res, "账号不存在", 400);

    const acc = accRows[0];
    const ok = await bcrypt.compare(password, acc.password_hash);
    if (!ok) return error(res, "密码错误", 400);

    const token = jwt.sign(
      { id: acc.id, username: acc.username, role: acc.role },
      JWT_SECRET,
      { expiresIn: "7d" }
    );

    // 前端管理员分支会判断 data.role === "admin"
    return success(
      res,
      { token, role: acc.role, username: acc.username },
      "登录成功"
    );
  } catch (err) {
    console.error("登录错误：", err);
    error(res, "服务器错误", 500);
  }
});

// ============ 获取当前用户资料 ============
router.get("/profile", verifyToken, async (req, res) => {
  try {
    const { role, student_no, username } = req.user || {};

    // 学生
    if (role === "student" && student_no) {
      const [rows] = await pool.query(
        `SELECT s.real_name, s.student_no, s.college, s.major, s.grade,
                d.building_no, d.room_no
         FROM student_profile s
         LEFT JOIN dorm_room d ON s.dorm_room_id = d.id
         WHERE s.student_no=?`,
        [student_no]
      );
      if (rows.length === 0) return error(res, "找不到用户", 404);
      return success(res, rows[0], "获取成功");
    }

    // 管理员仅返回基础信息（如需更多自己扩展）
    return success(res, { username, role }, "获取成功");
  } catch (err) {
    console.error("获取用户信息错误：", err);
    error(res, "服务器错误", 500);
  }
});

// ============ 更新学生资料 ============
router.put("/update", verifyToken, async (req, res) => {
  try {
    if (req.user?.role !== "student")
      return error(res, "仅学生可更新资料", 403);
    const { student_no } = req.user;
    const { real_name, college, major, grade } = req.body || {};

    const [ret] = await pool.query(
      "UPDATE student_profile SET real_name=?, college=?, major=?, grade=? WHERE student_no=?",
      [real_name, college, major, grade, student_no]
    );

    if (ret.affectedRows > 0) return success(res, null, "信息更新成功");
    return error(res, "未找到该学生", 404);
  } catch (err) {
    console.error("更新失败:", err);
    error(res, "服务器错误", 500);
  }
});

// === 二次确认口令（入库前使用）===
// POST /api/auth/confirm-password
// body: { password }
router.post("/confirm-password", verifyToken, async (req, res) => {
  try {
    const { password } = req.body || {};
    if (!password) return error(res, "缺少密码", 400);

    // 当前登录账号（JWT 里已有 id）
    const accountId = req.user?.id;
    if (!accountId) return error(res, "令牌缺少用户标识", 401);

    const [rows] = await pool.query(
      "SELECT id, password_hash FROM account WHERE id=? LIMIT 1",
      [accountId]
    );
    if (rows.length === 0) return error(res, "账号不存在", 404);

    const ok = await bcrypt.compare(password, rows[0].password_hash);
    if (!ok) return error(res, "密码不正确", 400);

    return success(res, null, "验证通过");
  } catch (err) {
    console.error("confirm-password error:", err);
    return error(res, "服务器错误", 500);
  }
});

export default router;
