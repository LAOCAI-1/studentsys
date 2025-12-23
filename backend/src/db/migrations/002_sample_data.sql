-- 可选的测试数据
INSERT INTO dorm_room (building_no, room_no) VALUES ('63', '525') ON DUPLICATE KEY UPDATE room_no=room_no;

-- 账号：学号 20250001 / 密码 123456
INSERT INTO account (username, password_hash, phone, role)
VALUES ('20250001', '$2b$10$1KQm5uQvHznqfYqU4nS5wOvt4O5i7lJz2sO/1yO1O3M7N2tZ4UeK6', '13900000000', 'student')
ON DUPLICATE KEY UPDATE phone=VALUES(phone);

INSERT INTO student_profile (account_id, real_name, student_no, college, major, grade, dorm_room_id)
SELECT a.id, '测试同学', '20250001', '电气学院', '自动化', '大二', d.id
FROM account a, dorm_room d
WHERE a.username='20250001' AND d.building_no='63' AND d.room_no='525'
ON DUPLICATE KEY UPDATE college='电气学院';
