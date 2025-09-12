-- PostgreSQL Schema for NEWS2-Live
-- Includes TimescaleDB extension for time-series data

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Patients table (PostgreSQL)
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    ward_id TEXT NOT NULL,
    bed_number TEXT NOT NULL,
    age INTEGER NOT NULL CHECK (age >= 0 AND age <= 150),
    is_copd_patient BOOLEAN NOT NULL DEFAULT FALSE,
    assigned_nurse_id TEXT NOT NULL,
    admission_date TIMESTAMPTZ NOT NULL,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Clinical flags
    is_palliative BOOLEAN NOT NULL DEFAULT FALSE,
    do_not_escalate BOOLEAN NOT NULL DEFAULT FALSE,
    oxygen_dependent BOOLEAN NOT NULL DEFAULT FALSE,
    -- Indexes
    CONSTRAINT valid_age CHECK (age >= 0 AND age <= 150)
);

-- Create indexes for patients table
CREATE INDEX idx_patients_ward_id ON patients(ward_id);
CREATE INDEX idx_patients_nurse_id ON patients(assigned_nurse_id);
CREATE INDEX idx_patients_admission_date ON patients(admission_date);

-- Vital Signs table (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS vital_signs (
    time TIMESTAMPTZ NOT NULL,
    event_id UUID NOT NULL DEFAULT uuid_generate_v4(),
    patient_id TEXT NOT NULL,
    -- Vital sign measurements
    respiratory_rate INTEGER NOT NULL CHECK (respiratory_rate >= 4 AND respiratory_rate <= 50),
    spo2 INTEGER NOT NULL CHECK (spo2 >= 50 AND spo2 <= 100),
    on_oxygen BOOLEAN NOT NULL DEFAULT FALSE,
    temperature NUMERIC(4,1) NOT NULL CHECK (temperature >= 30 AND temperature <= 45),
    systolic_bp INTEGER NOT NULL CHECK (systolic_bp >= 40 AND systolic_bp <= 300),
    heart_rate INTEGER NOT NULL CHECK (heart_rate >= 20 AND heart_rate <= 220),
    consciousness CHAR(1) NOT NULL CHECK (consciousness IN ('A', 'C', 'V', 'P', 'U')),
    -- Quality flags
    is_manual_entry BOOLEAN NOT NULL DEFAULT FALSE,
    has_artifacts BOOLEAN NOT NULL DEFAULT FALSE,
    confidence NUMERIC(3,2) NOT NULL DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    -- Constraints
    CONSTRAINT vital_signs_pkey PRIMARY KEY (time, patient_id),
    CONSTRAINT vital_signs_event_id_unique UNIQUE (event_id),
    CONSTRAINT vital_signs_patient_fkey FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

-- Convert to TimescaleDB hypertable with 1-hour chunks
SELECT create_hypertable('vital_signs', 'time', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE);

-- Create indexes for vital_signs
CREATE INDEX idx_vital_signs_patient_time ON vital_signs (patient_id, time DESC);
CREATE INDEX idx_vital_signs_event_id ON vital_signs (event_id);

-- Set compression policy for data older than 7 days
SELECT add_compression_policy('vital_signs', INTERVAL '7 days', if_not_exists => TRUE);

-- Set retention policy for 2 years
SELECT add_retention_policy('vital_signs', INTERVAL '2 years', if_not_exists => TRUE);

-- Audit Log table (Immutable)
CREATE TABLE IF NOT EXISTS audit_log (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    old_values JSONB,
    new_values JSONB,
    user_id TEXT NOT NULL,
    patient_id TEXT,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- Prevent updates and deletes
    CHECK (false) NO INHERIT
);

-- Create indexes for audit_log
CREATE INDEX idx_audit_log_timestamp ON audit_log(timestamp DESC);
CREATE INDEX idx_audit_log_patient_id ON audit_log(patient_id) WHERE patient_id IS NOT NULL;
CREATE INDEX idx_audit_log_table_operation ON audit_log(table_name, operation);

-- Create trigger function for audit logging
CREATE OR REPLACE FUNCTION audit_trigger_function() 
RETURNS TRIGGER AS $$
DECLARE
    audit_user_id TEXT;
    audit_patient_id TEXT;
BEGIN
    -- Get user_id from session variable or use system user
    audit_user_id := COALESCE(current_setting('app.user_id', true), session_user);
    
    -- Extract patient_id based on table
    IF TG_TABLE_NAME = 'patients' THEN
        audit_patient_id := CASE 
            WHEN TG_OP = 'DELETE' THEN OLD.patient_id
            ELSE NEW.patient_id
        END;
    ELSIF TG_TABLE_NAME = 'vital_signs' THEN
        audit_patient_id := CASE 
            WHEN TG_OP = 'DELETE' THEN OLD.patient_id
            ELSE NEW.patient_id
        END;
    END IF;
    
    -- Insert audit record
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (table_name, operation, new_values, user_id, patient_id)
        VALUES (TG_TABLE_NAME, TG_OP, to_jsonb(NEW), audit_user_id, audit_patient_id);
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (table_name, operation, old_values, new_values, user_id, patient_id)
        VALUES (TG_TABLE_NAME, TG_OP, to_jsonb(OLD), to_jsonb(NEW), audit_user_id, audit_patient_id);
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (table_name, operation, old_values, user_id, patient_id)
        VALUES (TG_TABLE_NAME, TG_OP, to_jsonb(OLD), audit_user_id, audit_patient_id);
    END IF;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create audit triggers for patients table
CREATE TRIGGER patients_audit_trigger
AFTER INSERT OR UPDATE OR DELETE ON patients
FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- Create audit triggers for vital_signs table
CREATE TRIGGER vital_signs_audit_trigger
AFTER INSERT OR UPDATE OR DELETE ON vital_signs
FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();

-- Prevent updates and deletes on audit_log table
CREATE OR REPLACE FUNCTION prevent_audit_modifications()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Audit log entries are immutable and cannot be modified or deleted';
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER audit_log_immutable_trigger
BEFORE UPDATE OR DELETE ON audit_log
FOR EACH ROW EXECUTE FUNCTION prevent_audit_modifications();

-- Create continuous aggregate for ward-level metrics (5-minute intervals)
CREATE MATERIALIZED VIEW IF NOT EXISTS ward_vital_signs_5min
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('5 minutes', vs.time) AS bucket,
    p.ward_id,
    COUNT(DISTINCT vs.patient_id) AS patient_count,
    AVG(vs.respiratory_rate) AS avg_respiratory_rate,
    AVG(vs.heart_rate) AS avg_heart_rate,
    AVG(vs.spo2) AS avg_spo2,
    AVG(vs.temperature) AS avg_temperature,
    AVG(vs.systolic_bp) AS avg_systolic_bp,
    SUM(CASE WHEN vs.consciousness != 'A' THEN 1 ELSE 0 END) AS altered_consciousness_count
FROM vital_signs vs
JOIN patients p ON vs.patient_id = p.patient_id
GROUP BY bucket, p.ward_id
WITH NO DATA;

-- Add refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('ward_vital_signs_5min',
    start_offset => INTERVAL '1 hour',
    end_offset => INTERVAL '5 minutes',
    schedule_interval => INTERVAL '5 minutes',
    if_not_exists => TRUE);