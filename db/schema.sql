CREATE TABLE IF NOT EXISTS students (
  enrolment_no        TEXT PRIMARY KEY,
  name                 TEXT,
  mobile               TEXT,
  mobile_verified      INTEGER,
  email                TEXT,
  email_verified       INTEGER,
  services             TEXT,   -- JSON array, mirrors MongoDB services[]
  verification         TEXT,   -- JSON array, heterogeneous (provost/chairman/pst_rejected/chm_rejected)
  reset_token          TEXT,
  reset_token_expires  TEXT
);