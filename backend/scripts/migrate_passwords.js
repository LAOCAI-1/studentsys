// backend/scripts/migrate_passwords.js
import dotenv from "dotenv";
import path from "path";
import { fileURLToPath } from "url";
import pool from "../src/db/pool.js";
import bcrypt from "bcrypt";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config({ path: path.join(__dirname, "..", ".env") }); // 读项目根 .env

const DB_NAME = process.env.DB_NAME || "studentsys";

async function tableExists(table) {
  const [rows] = await pool.query(
    "SELECT COUNT(*) AS c FROM information_schema.tables WHERE table_schema=? AND table_name=?",
    [DB_NAME, table]
  );
  return rows?.[0]?.c > 0;
}

async function columnExists(table, col) {
  const [rows] = await pool.query(
    "SELECT COUNT(*) AS c FROM information_schema.columns WHERE table_schema=? AND table_name=? AND column_name=?",
    [DB_NAME, table, col]
  );
  return rows?.[0]?.c > 0;
}

async function migrateUsers() {
  const [rows] = await pool.query(
    "SELECT id, username, password, password_hash FROM users WHERE (password_hash IS NULL OR password_hash='') AND (password IS NOT NULL AND password <> '')"
  );
  for (const u of rows) {
    const hash = await bcrypt.hash(u.password, 10);
    await pool.query("UPDATE users SET password_hash=? WHERE id=?", [
      hash,
      u.id,
    ]);
    console.log(`[users] migrated: ${u.username}`);
  }
  console.log(`[users] done, affected: ${rows.length}`);
}

async function migrateAccountIfLegacy() {
  const hasPlain = await columnExists("account", "password");
  const hasHash = await columnExists("account", "password_hash");
  if (!hasHash) {
    console.log(
      `[account] 没有 password_hash，说明你的 schema 不是当前版本，请先跑 001_schema_core.sql`
    );
    process.exit(1);
  }
  if (!hasPlain) {
    console.log(
      `[account] 没有明文 password 列，说明根本不需要迁移，直接退出。`
    );
    return;
  }
  const [rows] = await pool.query(
    "SELECT id, username, password, password_hash FROM account WHERE (password_hash IS NULL OR password_hash='') AND (password IS NOT NULL AND password <> '')"
  );
  for (const u of rows) {
    const hash = await bcrypt.hash(u.password, 10);
    await pool.query("UPDATE account SET password_hash=? WHERE id=?", [
      hash,
      u.id,
    ]);
    console.log(`[account] migrated: ${u.username}`);
  }
  console.log(`[account] done, affected: ${rows.length}`);
}

async function main() {
  const hasUsers = await tableExists("users");
  const hasAccount = await tableExists("account");

  if (hasUsers) {
    console.log("[migrate] 检测到 users 表，执行 users → password_hash 迁移");
    await migrateUsers();
    process.exit(0);
  }
  if (hasAccount) {
    console.log(
      "[migrate] 未检测到 users；检测到 account 表，尝试迁移 legacy account（若存在明文）"
    );
    await migrateAccountIfLegacy();
    process.exit(0);
  }
  console.log("[migrate] 既无 users 也无 account，退出（无需迁移）");
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
