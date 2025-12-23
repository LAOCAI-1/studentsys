USE studentsys;

CREATE TABLE IF NOT EXISTS face_identities (
  id          INT AUTO_INCREMENT PRIMARY KEY,
  account_id  INT NOT NULL,
  label       VARCHAR(100) NOT NULL,
  created_at  TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY uniq_acc_label (account_id, label),
  KEY idx_label (label),
  CONSTRAINT fk_face_ident_acc
    FOREIGN KEY (account_id) REFERENCES account(id)
    ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
