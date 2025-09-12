-- Initialize NEWS2 Live database
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create tables for patient data
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id VARCHAR(50) UNIQUE NOT NULL,
    ward_id VARCHAR(20) NOT NULL,
    bed_number VARCHAR(10),
    age INTEGER,
    is_copd_patient BOOLEAN DEFAULT FALSE,
    assigned_nurse_id VARCHAR(50),
    admission_date TIMESTAMP WITH TIME ZONE,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL,
    patient_id VARCHAR(50),
    event_data JSONB,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id VARCHAR(50)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_patients_patient_id ON patients(patient_id);
CREATE INDEX IF NOT EXISTS idx_patients_ward_id ON patients(ward_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_patient_id ON audit_log(patient_id);