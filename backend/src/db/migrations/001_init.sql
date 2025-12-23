-- === 选库（安全起见先创建再使用）===
CREATE DATABASE IF NOT EXISTS studentsys
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_0900_ai_ci;
USE studentsys;

-- === 基础三表 ===
CREATE TABLE IF NOT EXISTS account (
  id INT PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(100) NOT NULL UNIQUE,      -- 学号作用户名
  password_hash VARCHAR(255) NOT NULL,        -- bcrypt 60 也行，255 兼容将来升级
  phone VARCHAR(30),
  role ENUM('student','admin') NOT NULL DEFAULT 'student',
  created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS dorm_room (
  id INT PRIMARY KEY AUTO_INCREMENT,
  building_no VARCHAR(20) NOT NULL,
  room_no VARCHAR(20) NOT NULL,
  UNIQUE KEY uk_dorm (building_no, room_no)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS student_profile (
  id INT PRIMARY KEY AUTO_INCREMENT,
  account_id INT NOT NULL,
  real_name VARCHAR(50) NOT NULL,
  student_no VARCHAR(50) NOT NULL UNIQUE,
  college VARCHAR(100),
  major VARCHAR(100),
  grade VARCHAR(20),
  dorm_room_id INT,
  created_at TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT fk_sp_acc  FOREIGN KEY (account_id)  REFERENCES account(id)   ON DELETE CASCADE,
  CONSTRAINT fk_sp_dorm FOREIGN KEY (dorm_room_id) REFERENCES dorm_room(id) ON DELETE SET NULL,
  KEY idx_sp_account (account_id),
  KEY idx_sp_dorm    (dorm_room_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- === 人脸识别日志（供 /api/face/log 使用）===
CREATE TABLE IF NOT EXISTS face_logs (
  id BIGINT PRIMARY KEY AUTO_INCREMENT,
  label VARCHAR(100) NOT NULL,     -- 识别到的姓名/ID
  score DOUBLE NOT NULL,           -- 置信度
  threshold DOUBLE NOT NULL,       -- 判定阈值（记录入库时的阈值）
  source ENUM('upload','camera') NOT NULL,
  raw_json JSON NOT NULL,          -- 原始返回，便于审计/回溯
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  KEY idx_fl_label (label),
  KEY idx_fl_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
