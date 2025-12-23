-- 账号 ↔ 人脸标签 绑定关系
CREATE TABLE IF NOT EXISTS `face_identities` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `account_id` INT NOT NULL,
  `label` VARCHAR(100) NOT NULL,
  `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `uniq_acc_label` (`account_id`, `label`),
  KEY `idx_label` (`label`),
  CONSTRAINT `fk_face_ident_acc` FOREIGN KEY (`account_id`)
    REFERENCES `account`(`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 若你还没建 face_logs（之前代码会用到），也一并补上：
CREATE TABLE IF NOT EXISTS `face_logs` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `label` VARCHAR(100) NOT NULL,
  `score` DOUBLE NOT NULL DEFAULT 0,
  `threshold` DOUBLE NOT NULL DEFAULT 0.55,
  `source` VARCHAR(32) DEFAULT 'upload',
  `raw_json` JSON NULL,
  `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `idx_label` (`label`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
