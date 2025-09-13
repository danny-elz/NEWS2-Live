// Core NEWS2 and Alert Types
export interface Patient {
  id: string;
  name: string;
  mrn: string; // Medical Record Number
  dateOfBirth: string;
  gender: 'male' | 'female' | 'other';
  ward: string;
  bed: string;
  admissionDate: string;
  currentNEWS2Score: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
}

export interface VitalSigns {
  respiratoryRate: number;
  oxygenSaturation: number;
  supplementalOxygen: boolean;
  temperature: number;
  systolicBloodPressure: number;
  heartRate: number;
  consciousnessLevel: 'alert' | 'voice' | 'pain' | 'unresponsive';
  timestamp: string;
}

export interface NEWS2Score {
  id: string;
  patientId: string;
  vitalSigns: VitalSigns;
  score: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  timestamp: string;
  calculatedBy: string;
}

export interface Alert {
  id: string;
  patientId: string;
  patient: Patient;
  type: 'news2_threshold' | 'vital_critical' | 'trend_deterioration' | 'system_error';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  message: string;
  timestamp: string;
  acknowledged: boolean;
  acknowledgedBy?: string;
  acknowledgedAt?: string;
  resolved: boolean;
  resolvedBy?: string;
  resolvedAt?: string;
  triggerValue?: number;
  threshold?: number;
  relatedScoreId?: string;
  suppressedUntil?: string;
  suppressionReason?: string;
}

export interface SuppressionRule {
  id: string;
  name: string;
  description: string;
  conditions: SuppressionCondition[];
  duration: number; // in minutes
  priority: number;
  active: boolean;
  createdBy: string;
  createdAt: string;
  lastModified: string;
  appliedCount: number;
}

export interface SuppressionCondition {
  field: string;
  operator: 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte' | 'in' | 'not_in';
  value: any;
  logicalOperator?: 'and' | 'or';
}

export interface DashboardMetrics {
  totalPatients: number;
  activeAlerts: number;
  criticalAlerts: number;
  suppressedAlerts: number;
  averageNEWS2Score: number;
  alertTrends: {
    timestamp: string;
    alertCount: number;
    suppressedCount: number;
  }[];
  systemHealth: {
    status: 'healthy' | 'warning' | 'critical';
    uptime: number;
    lastUpdate: string;
    services: {
      name: string;
      status: 'online' | 'offline' | 'degraded';
      lastCheck: string;
    }[];
  };
}

// User and Authentication Types
export interface User {
  id: string;
  username: string;
  email: string;
  firstName: string;
  lastName: string;
  role: UserRole;
  department: string;
  isActive: boolean;
  lastLogin?: string;
  preferences: UserPreferences;
}

export type UserRole = 'nurse' | 'doctor' | 'admin' | 'viewer';

export interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  notifications: {
    email: boolean;
    push: boolean;
    sound: boolean;
  };
  dashboard: {
    refreshInterval: number;
    defaultView: string;
  };
}

export interface AuthState {
  isAuthenticated: boolean;
  user: User | null;
  token: string | null;
  loading: boolean;
  error: string | null;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  totalPages: number;
}

export interface ErrorResponse {
  success: false;
  error: {
    code: string;
    message: string;
    details?: any;
  };
  timestamp: string;
}

// WebSocket Types
export interface WebSocketMessage {
  type: 'alert' | 'suppression' | 'metrics_update' | 'patient_update' | 'system_status';
  payload: any;
  timestamp: string;
}

export interface AlertWebSocketMessage extends WebSocketMessage {
  type: 'alert';
  payload: {
    alert: Alert;
    action: 'created' | 'updated' | 'acknowledged' | 'resolved' | 'suppressed';
  };
}

// Form Types
export interface AlertFilters {
  severity?: string[];
  type?: string[];
  acknowledged?: boolean;
  resolved?: boolean;
  patientId?: string;
  dateRange?: {
    start: string;
    end: string;
  };
}

export interface SuppressionRuleForm {
  name: string;
  description: string;
  conditions: SuppressionCondition[];
  duration: number;
  priority: number;
  active: boolean;
}

// Chart and Visualization Types
export interface ChartDataPoint {
  timestamp: string;
  value: number;
  label?: string;
}

export interface TrendData {
  patientId: string;
  metric: string;
  data: ChartDataPoint[];
  threshold?: number;
  unit: string;
}

// Configuration Types
export interface SystemConfig {
  alertThresholds: {
    news2: {
      medium: number;
      high: number;
      critical: number;
    };
    vitalSigns: {
      [key: string]: {
        min?: number;
        max?: number;
      };
    };
  };
  suppressionSettings: {
    maxDuration: number;
    defaultDuration: number;
    requireApproval: boolean;
  };
  notifications: {
    emailEnabled: boolean;
    smsEnabled: boolean;
    escalationDelay: number;
  };
}