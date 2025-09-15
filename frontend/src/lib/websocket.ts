import { Patient } from '@/components/dashboard/patient-status-grid';
import { Alert } from '@/components/dashboard/alert-summary-panel';

export interface VitalsUpdate {
  patient_id: string;
  ts: string;
  resp_rate: number;
  spo2: number;
  o2_supplemental: boolean;
  temp_c: number;
  sbp: number;
  hr: number;
  avpu: 'A' | 'V' | 'P' | 'U';
  source_seq: number;
}

export interface NEWS2Update {
  patient_id: string;
  ts: string;
  total: number;
  component_scores: {
    resp: number;
    spo2: number;
    o2: number;
    temp: number;
    sbp: number;
    hr: number;
    avpu: number;
  };
  hard_flag: boolean;
  single_param_eq3: boolean;
}

export interface AlertData {
  alert_id: string;
  patient_id: string;
  ts: string;
  news2: number;
  reasons: string[];
  acknowledged: boolean;
}

export interface WebSocketMessage {
  type: 'vitals_update' | 'news2_update' | 'alert_created' | 'system_status' | 'patient_update' | 'connection_status';
  data?: any;
  payload?: any;
  timestamp?: string;
}

export class NEWS2WebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectInterval = 3000;
  private isConnected = false;

  private eventHandlers: { [key: string]: Function[] } = {
    vitals_update: [],
    news2_update: [],
    alert_created: [],
    patient_update: [],
    connection_status: [],
    system_status: [],
    connect: [],
    disconnect: [],
    error: []
  };

  constructor(private url: string = 'ws://localhost:8765') {}

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log('🔗 WebSocket connected to NEWS2 backend');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.emit('connect');
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            console.log('📨 WebSocket message:', message);
            this.handleMessage(message);
          } catch (error) {
            console.error('❌ Error parsing WebSocket message:', error);
          }
        };

        this.ws.onclose = () => {
          console.log('🔌 WebSocket disconnected');
          this.isConnected = false;
          this.emit('disconnect');
          this.attemptReconnect();
        };

        this.ws.onerror = (error) => {
          console.error('❌ WebSocket error:', error);
          this.emit('error', error);
          reject(error);
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  private handleMessage(message: WebSocketMessage) {
    const { type } = message;

    switch (type) {
      case 'vitals_update':
        this.emit('vitals_update', message.data as VitalsUpdate);
        break;
      case 'news2_update':
        this.emit('news2_update', message.data as NEWS2Update);
        break;
      case 'alert_created':
        this.emit('alert_created', message.data as AlertData);
        break;
      case 'patient_update':
        // Handle both formats - data directly or in payload
        const patientData = message.data || message.payload?.data;
        this.emit('patient_update', patientData);
        break;
      case 'system_status':
      case 'connection_status':
        this.emit(type, message.data || message);
        break;
      default:
        console.log('🔍 Unknown message type:', type);
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('❌ Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    console.log(`🔄 Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);

    setTimeout(() => {
      this.connect().catch(error => {
        console.error('❌ Reconnection failed:', error);
      });
    }, this.reconnectInterval);
  }

  on(event: string, handler: Function) {
    if (!this.eventHandlers[event]) {
      this.eventHandlers[event] = [];
    }
    this.eventHandlers[event].push(handler);
  }

  off(event: string, handler: Function) {
    if (this.eventHandlers[event]) {
      this.eventHandlers[event] = this.eventHandlers[event].filter(h => h !== handler);
    }
  }

  private emit(event: string, data?: any) {
    if (this.eventHandlers[event]) {
      this.eventHandlers[event].forEach(handler => handler(data));
    }
  }

  send(message: any): boolean {
    if (this.ws && this.isConnected) {
      try {
        this.ws.send(JSON.stringify(message));
        return true;
      } catch (error) {
        console.error('❌ Error sending message:', error);
        return false;
      }
    }
    return false;
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  getConnectionStatus(): boolean {
    return this.isConnected;
  }

  // Convenience methods for sending specific messages
  subscribe(wardIds: string[] = ['all']) {
    return this.send({
      type: 'subscribe',
      ward_ids: wardIds
    });
  }

  sendHeartbeat() {
    return this.send({
      type: 'heartbeat',
      timestamp: new Date().toISOString()
    });
  }
}

// Helper functions to convert backend data to frontend format
export function vitalsToPatient(vitalsUpdate: VitalsUpdate, existingPatient?: Partial<Patient>): Partial<Patient> {
  const getRiskLevel = (news2Score: number) => {
    if (news2Score >= 7) return 'critical';
    if (news2Score >= 5) return 'high';
    if (news2Score >= 3) return 'medium';
    return 'low';
  };

  // Calculate approximate NEWS2 score from vitals (simplified)
  const approximateNEWS2 = Math.min(15, Math.floor(
    (vitalsUpdate.resp_rate > 20 ? 2 : 0) +
    (vitalsUpdate.spo2 < 94 ? 3 : vitalsUpdate.spo2 < 96 ? 2 : 0) +
    (vitalsUpdate.temp_c > 38.0 || vitalsUpdate.temp_c < 36.0 ? 1 : 0) +
    (vitalsUpdate.sbp < 110 ? 2 : 0) +
    (vitalsUpdate.hr > 100 || vitalsUpdate.hr < 60 ? 1 : 0) +
    (vitalsUpdate.avpu !== 'A' ? 3 : 0)
  ));

  return {
    ...existingPatient,
    id: vitalsUpdate.patient_id,
    news2Score: approximateNEWS2,
    riskLevel: getRiskLevel(approximateNEWS2) as 'low' | 'medium' | 'high' | 'critical',
    vitals: {
      heartRate: vitalsUpdate.hr,
      bloodPressure: { systolic: vitalsUpdate.sbp, diastolic: Math.floor(vitalsUpdate.sbp * 0.7) },
      temperature: vitalsUpdate.temp_c,
      respiratoryRate: vitalsUpdate.resp_rate,
      oxygenSaturation: vitalsUpdate.spo2,
      avpu: vitalsUpdate.avpu
    },
    lastUpdated: new Date(vitalsUpdate.ts).toLocaleTimeString(),
  };
}

export function news2UpdateToPatient(news2Update: NEWS2Update, existingPatient?: Partial<Patient>): Partial<Patient> {
  const getRiskLevel = (score: number) => {
    if (score >= 7) return 'critical';
    if (score >= 5) return 'high';
    if (score >= 3) return 'medium';
    return 'low';
  };

  return {
    ...existingPatient,
    id: news2Update.patient_id,
    news2Score: news2Update.total,
    riskLevel: getRiskLevel(news2Update.total) as 'low' | 'medium' | 'high' | 'critical',
    lastUpdated: new Date(news2Update.ts).toLocaleTimeString(),
  };
}

export function alertDataToAlert(alertData: AlertData): Alert {
  return {
    id: alertData.alert_id,
    patientId: alertData.patient_id,
    patientName: `Patient ${alertData.patient_id}`,
    bedNumber: 'TBD', // Would come from patient data
    wardLocation: 'ICU', // Would come from patient data
    alertType: 'news2',
    priority: alertData.news2 >= 7 ? 'critical' : 'high',
    title: `NEWS2 Score ${alertData.news2}`,
    description: alertData.reasons.join(', '),
    timestamp: new Date(alertData.ts).toLocaleTimeString(),
    acknowledged: alertData.acknowledged,
    escalated: false
  };
}