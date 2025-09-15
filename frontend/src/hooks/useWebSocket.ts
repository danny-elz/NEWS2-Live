import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import {
  NEWS2WebSocketClient,
  VitalsUpdate,
  NEWS2Update,
  AlertData,
  vitalsToPatient,
  news2UpdateToPatient,
  alertDataToAlert
} from '@/lib/websocket';
import { Patient } from '@/components/dashboard/patient-status-grid';
import { Alert } from '@/components/dashboard/alert-summary-panel';

interface WebSocketState {
  isConnected: boolean;
  connectionStatus: string;
  error: string | null;
}

export function useWebSocket() {
  const clientRef = useRef<NEWS2WebSocketClient | null>(null);
  const [state, setState] = useState<WebSocketState>({
    isConnected: false,
    connectionStatus: 'disconnected',
    error: null
  });

  // Initialize WebSocket client
  useEffect(() => {
    clientRef.current = new NEWS2WebSocketClient();

    // Set up event handlers
    clientRef.current.on('connect', () => {
      setState(prev => ({
        ...prev,
        isConnected: true,
        connectionStatus: 'connected',
        error: null
      }));

      // Subscribe to all wards
      clientRef.current?.subscribe(['all']);
    });

    clientRef.current.on('disconnect', () => {
      setState(prev => ({
        ...prev,
        isConnected: false,
        connectionStatus: 'disconnected'
      }));
    });

    clientRef.current.on('error', (error: any) => {
      setState(prev => ({
        ...prev,
        error: error?.message || 'Connection error',
        connectionStatus: 'error'
      }));
    });

    // Connect to the server
    clientRef.current.connect().catch(error => {
      console.error('Failed to connect to WebSocket server:', error);
      setState(prev => ({
        ...prev,
        error: 'Failed to connect to server',
        connectionStatus: 'error'
      }));
    });

    // Cleanup on unmount
    return () => {
      clientRef.current?.disconnect();
    };
  }, []);

  const sendMessage = useCallback((message: any) => {
    return clientRef.current?.send(message) || false;
  }, []);

  const subscribe = useCallback((wardIds: string[] = ['all']) => {
    return clientRef.current?.subscribe(wardIds) || false;
  }, []);

  return {
    ...state,
    sendMessage,
    subscribe,
    client: clientRef.current
  };
}

export function usePatientUpdates() {
  const [patients, setPatients] = useState<{ [id: string]: Patient }>({});
  const clientRef = useRef<NEWS2WebSocketClient | null>(null);

  useEffect(() => {
    clientRef.current = new NEWS2WebSocketClient();

    clientRef.current.on('vitals_update', (vitalsUpdate: VitalsUpdate) => {
      setPatients(prev => {
        const existingPatient = prev[vitalsUpdate.patient_id];
        const updatedPatient = vitalsToPatient(vitalsUpdate, existingPatient);

        return {
          ...prev,
          [vitalsUpdate.patient_id]: {
            ...existingPatient,
            ...updatedPatient,
            // Ensure required fields have defaults
            name: existingPatient?.name || `Patient ${vitalsUpdate.patient_id}`,
            age: existingPatient?.age || 45,
            bedNumber: existingPatient?.bedNumber || `B${vitalsUpdate.patient_id.slice(-2)}`,
            wardLocation: existingPatient?.wardLocation || 'ICU',
            alerts: existingPatient?.alerts || 0,
            assignedNurse: existingPatient?.assignedNurse || 'Staff Nurse'
          } as Patient
        };
      });
    });

    clientRef.current.on('news2_update', (news2Update: NEWS2Update) => {
      setPatients(prev => {
        const existingPatient = prev[news2Update.patient_id];
        const updatedPatient = news2UpdateToPatient(news2Update, existingPatient);

        return {
          ...prev,
          [news2Update.patient_id]: {
            ...existingPatient,
            ...updatedPatient,
            // Ensure required fields have defaults
            name: existingPatient?.name || `Patient ${news2Update.patient_id}`,
            age: existingPatient?.age || 45,
            bedNumber: existingPatient?.bedNumber || `B${news2Update.patient_id.slice(-2)}`,
            wardLocation: existingPatient?.wardLocation || 'ICU',
            alerts: existingPatient?.alerts || 0,
            assignedNurse: existingPatient?.assignedNurse || 'Staff Nurse',
            vitals: existingPatient?.vitals || {
              heartRate: 75,
              bloodPressure: { systolic: 120, diastolic: 80 },
              temperature: 36.8,
              respiratoryRate: 16,
              oxygenSaturation: 98,
              avpu: 'A'
            }
          } as Patient
        };
      });
    });

    clientRef.current.on('patient_update', (patientData: any) => {
      // Handle the comprehensive patient update format from simple_websocket_server.py
      if (patientData) {
        const patient: Patient = {
          id: patientData.id || patientData.patient_id,
          name: patientData.name || `Patient ${patientData.id}`,
          age: patientData.age || 45,
          bedNumber: patientData.bed || patientData.bedNumber || `B${patientData.id}`,
          wardLocation: patientData.ward || patientData.wardLocation || 'ICU',
          news2Score: patientData.currentNEWS2Score || patientData.news2Score || 0,
          riskLevel: patientData.riskLevel as 'low' | 'medium' | 'high' | 'critical',
          vitals: patientData.vitals || {
            heartRate: 75,
            bloodPressure: { systolic: 120, diastolic: 80 },
            temperature: 36.8,
            respiratoryRate: 16,
            oxygenSaturation: 98,
            avpu: 'A'
          },
          lastUpdated: new Date(patientData.lastUpdated || Date.now()).toLocaleTimeString(),
          alerts: patientData.alertCount || patientData.alerts || 0,
          assignedNurse: patientData.assignedNurse || 'Staff Nurse'
        };

        setPatients(prev => ({
          ...prev,
          [patient.id]: patient
        }));
      }
    });

    clientRef.current.connect().catch(console.error);

    return () => {
      clientRef.current?.disconnect();
    };
  }, []);

  const patientsArray = useMemo(() => Object.values(patients), [patients]);

  return {
    patients: patientsArray,
    patientsById: patients
  };
}

export function useAlertUpdates() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const clientRef = useRef<NEWS2WebSocketClient | null>(null);

  useEffect(() => {
    clientRef.current = new NEWS2WebSocketClient();

    clientRef.current.on('alert_created', (alertData: AlertData) => {
      const newAlert = alertDataToAlert(alertData);
      setAlerts(prev => [newAlert, ...prev]);
    });

    clientRef.current.connect().catch(console.error);

    return () => {
      clientRef.current?.disconnect();
    };
  }, []);

  const acknowledgeAlert = useCallback((alertId: string) => {
    setAlerts(prev => prev.map(alert =>
      alert.id === alertId
        ? { ...alert, acknowledged: true, assignedTo: 'Current User' }
        : alert
    ));
  }, []);

  return {
    alerts,
    acknowledgeAlert
  };
}

export function useSystemStatus() {
  const [systemStatus, setSystemStatus] = useState({
    server: 'offline' as 'online' | 'offline' | 'degraded',
    database: 'offline' as 'online' | 'offline' | 'degraded',
    websocket: 'offline' as 'online' | 'offline' | 'degraded',
    integration: 'offline' as 'online' | 'offline' | 'degraded'
  });

  const clientRef = useRef<NEWS2WebSocketClient | null>(null);

  useEffect(() => {
    clientRef.current = new NEWS2WebSocketClient();

    clientRef.current.on('connect', () => {
      setSystemStatus(prev => ({
        ...prev,
        websocket: 'online',
        server: 'online'
      }));
    });

    clientRef.current.on('disconnect', () => {
      setSystemStatus(prev => ({
        ...prev,
        websocket: 'offline',
        server: 'offline'
      }));
    });

    clientRef.current.on('system_status', (status: any) => {
      if (status?.data?.status === 'healthy') {
        setSystemStatus(prev => ({
          ...prev,
          server: 'online',
          database: 'online',
          integration: 'online'
        }));
      }
    });

    clientRef.current.connect().catch(() => {
      setSystemStatus(prev => ({
        ...prev,
        websocket: 'offline',
        server: 'offline'
      }));
    });

    return () => {
      clientRef.current?.disconnect();
    };
  }, []);

  return systemStatus;
}