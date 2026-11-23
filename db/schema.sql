-- Mock student DB, shaped like the real MongoDB student document
-- (schemastudent.json). Nested parts are stored as JSON text so the
-- normalisation step in student_tools.py is tested against realistic shapes.
DROP TABLE IF EXISTS students;

CREATE TABLE students (
  enroll               TEXT PRIMARY KEY,   -- real field: "enroll"
  full_name            TEXT,               -- real field: "fullName"
  email                TEXT,
  mobile               TEXT,
  profile              TEXT,               -- JSON: verification[], hall, department, ...
  contact              TEXT,               -- JSON: emailID.isVerified, mobileID.isVerified
  services             TEXT,               -- JSON array: one entry per service (WiFi, Email, ...)
  reset_token          TEXT,
  reset_token_expires  TEXT
);
