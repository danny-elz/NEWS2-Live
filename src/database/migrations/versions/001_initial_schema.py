"""Initial schema with patients, vital_signs, and audit_log tables

Revision ID: 001
Revises: 
Create Date: 2024-09-11 23:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Enable required extensions
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute('CREATE EXTENSION IF NOT EXISTS timescaledb')
    
    # Create patients table
    op.create_table(
        'patients',
        sa.Column('patient_id', sa.Text(), nullable=False),
        sa.Column('ward_id', sa.Text(), nullable=False),
        sa.Column('bed_number', sa.Text(), nullable=False),
        sa.Column('age', sa.Integer(), nullable=False),
        sa.Column('is_copd_patient', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('assigned_nurse_id', sa.Text(), nullable=False),
        sa.Column('admission_date', sa.DateTime(timezone=True), nullable=False),
        sa.Column('last_updated', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('is_palliative', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('do_not_escalate', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('oxygen_dependent', sa.Boolean(), nullable=False, server_default='false'),
        sa.CheckConstraint('age >= 0 AND age <= 150', name='valid_age'),
        sa.PrimaryKeyConstraint('patient_id')
    )
    
    # Create indexes for patients table
    op.create_index('idx_patients_ward_id', 'patients', ['ward_id'])
    op.create_index('idx_patients_nurse_id', 'patients', ['assigned_nurse_id'])
    op.create_index('idx_patients_admission_date', 'patients', ['admission_date'])
    
    # Create vital_signs table
    op.create_table(
        'vital_signs',
        sa.Column('time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('event_id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('patient_id', sa.Text(), nullable=False),
        sa.Column('respiratory_rate', sa.Integer(), nullable=False),
        sa.Column('spo2', sa.Integer(), nullable=False),
        sa.Column('on_oxygen', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('temperature', sa.Numeric(4, 1), nullable=False),
        sa.Column('systolic_bp', sa.Integer(), nullable=False),
        sa.Column('heart_rate', sa.Integer(), nullable=False),
        sa.Column('consciousness', sa.CHAR(1), nullable=False),
        sa.Column('is_manual_entry', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('has_artifacts', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('confidence', sa.Numeric(3, 2), nullable=False, server_default='1.0'),
        sa.CheckConstraint('respiratory_rate >= 4 AND respiratory_rate <= 50', name='vital_signs_respiratory_rate_check'),
        sa.CheckConstraint('spo2 >= 50 AND spo2 <= 100', name='vital_signs_spo2_check'),
        sa.CheckConstraint('temperature >= 30 AND temperature <= 45', name='vital_signs_temperature_check'),
        sa.CheckConstraint('systolic_bp >= 40 AND systolic_bp <= 300', name='vital_signs_systolic_bp_check'),
        sa.CheckConstraint('heart_rate >= 20 AND heart_rate <= 220', name='vital_signs_heart_rate_check'),
        sa.CheckConstraint("consciousness IN ('A', 'C', 'V', 'P', 'U')", name='vital_signs_consciousness_check'),
        sa.CheckConstraint('confidence >= 0 AND confidence <= 1', name='vital_signs_confidence_check'),
        sa.ForeignKeyConstraint(['patient_id'], ['patients.patient_id'], ),
        sa.PrimaryKeyConstraint('time', 'patient_id'),
        sa.UniqueConstraint('event_id', name='vital_signs_event_id_unique')
    )
    
    # Convert to TimescaleDB hypertable
    op.execute("SELECT create_hypertable('vital_signs', 'time', chunk_time_interval => INTERVAL '1 hour', if_not_exists => TRUE)")
    
    # Create indexes for vital_signs
    op.create_index('idx_vital_signs_patient_time', 'vital_signs', ['patient_id', sa.text('time DESC')])
    op.create_index('idx_vital_signs_event_id', 'vital_signs', ['event_id'])
    
    # Set compression and retention policies
    op.execute("SELECT add_compression_policy('vital_signs', INTERVAL '7 days', if_not_exists => TRUE)")
    op.execute("SELECT add_retention_policy('vital_signs', INTERVAL '2 years', if_not_exists => TRUE)")
    
    # Create audit_log table
    op.create_table(
        'audit_log',
        sa.Column('audit_id', postgresql.UUID(as_uuid=True), nullable=False, server_default=sa.text('uuid_generate_v4()')),
        sa.Column('table_name', sa.Text(), nullable=False),
        sa.Column('operation', sa.Text(), nullable=False),
        sa.Column('old_values', postgresql.JSONB(astext_type=sa.Text())),
        sa.Column('new_values', postgresql.JSONB(astext_type=sa.Text())),
        sa.Column('user_id', sa.Text(), nullable=False),
        sa.Column('patient_id', sa.Text()),
        sa.Column('timestamp', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('NOW()')),
        sa.CheckConstraint("operation IN ('INSERT', 'UPDATE', 'DELETE')", name='audit_log_operation_check'),
        sa.PrimaryKeyConstraint('audit_id')
    )
    
    # Create indexes for audit_log
    op.create_index('idx_audit_log_timestamp', 'audit_log', [sa.text('timestamp DESC')])
    op.create_index('idx_audit_log_patient_id', 'audit_log', ['patient_id'])
    op.create_index('idx_audit_log_table_operation', 'audit_log', ['table_name', 'operation'])
    
    # Create audit trigger function
    op.execute("""
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
    """)
    
    # Create audit triggers
    op.execute("""
    CREATE TRIGGER patients_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON patients
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
    """)
    
    op.execute("""
    CREATE TRIGGER vital_signs_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON vital_signs
    FOR EACH ROW EXECUTE FUNCTION audit_trigger_function();
    """)
    
    # Prevent modifications to audit_log
    op.execute("""
    CREATE OR REPLACE FUNCTION prevent_audit_modifications()
    RETURNS TRIGGER AS $$
    BEGIN
        RAISE EXCEPTION 'Audit log entries are immutable and cannot be modified or deleted';
        RETURN NULL;
    END;
    $$ LANGUAGE plpgsql;
    """)
    
    op.execute("""
    CREATE TRIGGER audit_log_immutable_trigger
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION prevent_audit_modifications();
    """)
    
    # Create continuous aggregate for ward-level metrics
    op.execute("""
    CREATE MATERIALIZED VIEW ward_vital_signs_5min
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
    """)
    
    # Add refresh policy for continuous aggregate
    op.execute("""
    SELECT add_continuous_aggregate_policy('ward_vital_signs_5min',
        start_offset => INTERVAL '1 hour',
        end_offset => INTERVAL '5 minutes',
        schedule_interval => INTERVAL '5 minutes',
        if_not_exists => TRUE);
    """)


def downgrade() -> None:
    # Drop continuous aggregate and its policy
    op.execute("DROP MATERIALIZED VIEW IF EXISTS ward_vital_signs_5min")
    
    # Drop triggers
    op.execute("DROP TRIGGER IF EXISTS audit_log_immutable_trigger ON audit_log")
    op.execute("DROP TRIGGER IF EXISTS vital_signs_audit_trigger ON vital_signs")
    op.execute("DROP TRIGGER IF EXISTS patients_audit_trigger ON patients")
    
    # Drop functions
    op.execute("DROP FUNCTION IF EXISTS prevent_audit_modifications()")
    op.execute("DROP FUNCTION IF EXISTS audit_trigger_function()")
    
    # Drop tables
    op.drop_table('audit_log')
    op.drop_table('vital_signs')
    op.drop_table('patients')
    
    # Drop extensions (be careful - other applications might need them)
    # op.execute('DROP EXTENSION IF EXISTS timescaledb')
    # op.execute('DROP EXTENSION IF EXISTS "uuid-ossp"')