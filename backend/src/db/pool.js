import mysql from "mysql2/promise";
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
// .env 放在项目根目录 studentsys/.env
dotenv.config({ path: path.resolve(__dirname, "../../../.env") });

const pool = mysql.createPool({
  host: process.env.DB_HOST || "127.0.0.1",
  port: Number(process.env.DB_PORT || 3306),
  user: process.env.DB_USER || "root",
  password: process.env.DB_PASSWORD || "",
  database: process.env.DB_NAME || "studentsys",
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
});

export default pool;
