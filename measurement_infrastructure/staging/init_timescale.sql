-- Initialize TimescaleDB for vital signs data
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create vital signs table
CREATE TABLE IF NOT EXISTS vital_signs (
    time TIMESTAMPTZ NOT NULL,
    patient_id TEXT NOT NULL,
    event_id UUID,
    respiratory_rate INTEGER,
    spo2 INTEGER,
    on_oxygen BOOLEAN,
    temperature DECIMAL(4,1),
    systolic_bp INTEGER,
    heart_rate INTEGER,
    consciousness VARCHAR(20),
    data_source VARCHAR(50),
    quality_flags JSONB
);

-- Convert to hypertable
SELECT create_hypertable('vital_signs', 'time', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_vital_signs_patient_id ON vital_signs(patient_id, time DESC);
CREATE INDEX IF NOT EXISTS idx_vital_signs_event_id ON vital_signs(event_id);

-- Create NEWS2 results table
CREATE TABLE IF NOT EXISTS news2_results (
    time TIMESTAMPTZ NOT NULL,
    patient_id TEXT NOT NULL,
    event_id UUID,
    news2_score INTEGER,
    individual_scores JSONB,
    risk_category VARCHAR(20),
    scale_used INTEGER,
    red_flags TEXT[],
    calculation_time_ms DECIMAL(8,2)
);

SELECT create_hypertable('news2_results', 'time', if_not_exists => TRUE);

-- Create retention policies
SELECT add_retention_policy('vital_signs', INTERVAL '30 days', if_not_exists => TRUE);
SELECT add_retention_policy('news2_results', INTERVAL '30 days', if_not_exists => TRUE);

-- Create compression policies  
SELECT add_compression_policy('vital_signs', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('news2_results', INTERVAL '7 days', if_not_exists => TRUE);