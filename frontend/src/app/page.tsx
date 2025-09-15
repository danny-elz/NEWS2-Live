'use client';

import { useState, useEffect, useMemo } from 'react';
import { SimpleNavbar } from '@/components/layout/simple-navbar';
import { QuickActionsDock } from '@/components/layout/quick-actions-dock';
import { SystemStatus } from '@/components/layout/system-status';
import { PatientStatusGrid, type Patient } from '@/components/dashboard/patient-status-grid';
import { AlertSummaryPanel, type Alert } from '@/components/dashboard/alert-summary-panel';
import { usePatientUpdates, useAlertUpdates, useSystemStatus, useWebSocket } from '@/hooks/useWebSocket';
import { Badge } from '@/components/ui/badge';

// Mock data for demonstration
const mockPatients: Patient[] = [
  {
    id: '1',
    name: 'John Smith',
    age: 65,
    bedNumber: '101',
    wardLocation: 'ICU',
    news2Score: 12,
    riskLevel: 'critical',
    vitals: {
      heartRate: 110,
      bloodPressure: { systolic: 160, diastolic: 90 },
      temperature: 38.5,
      respiratoryRate: 22,
      oxygenSaturation: 92,
      avpu: 'V'
    },
    lastUpdated: '5 min ago',
    alerts: 2,
    assignedNurse: 'Sarah Johnson'
  },
  {
    id: '2',
    name: 'Mary Davis',
    age: 72,
    bedNumber: '102',
    wardLocation: 'General Ward',
    news2Score: 6,
    riskLevel: 'medium',
    vitals: {
      heartRate: 88,
      bloodPressure: { systolic: 140, diastolic: 80 },
      temperature: 37.2,
      respiratoryRate: 18,
      oxygenSaturation: 96,
      avpu: 'A'
    },
    lastUpdated: '12 min ago',
    alerts: 0,
    assignedNurse: 'Mike Wilson'
  },
  {
    id: '3',
    name: 'Robert Brown',
    age: 58,
    bedNumber: '201',
    wardLocation: 'Cardiology',
    news2Score: 2,
    riskLevel: 'low',
    vitals: {
      heartRate: 75,
      bloodPressure: { systolic: 120, diastolic: 70 },
      temperature: 36.8,
      respiratoryRate: 16,
      oxygenSaturation: 98,
      avpu: 'A'
    },
    lastUpdated: '8 min ago',
    alerts: 0,
    assignedNurse: 'Lisa Chen'
  }
];

const mockAlerts: Alert[] = [
  {
    id: '1',
    patientId: '1',
    patientName: 'John Smith',
    bedNumber: '101',
    wardLocation: 'ICU',
    alertType: 'news2',
    priority: 'critical',
    title: 'NEWS2 Score Critical',
    description: 'NEWS2 score has reached 12 indicating critical deterioration',
    timestamp: '2 min ago',
    acknowledged: false,
    escalated: true
  },
  {
    id: '2',
    patientId: '1',
    patientName: 'John Smith',
    bedNumber: '101',
    wardLocation: 'ICU',
    alertType: 'vitals',
    priority: 'high',
    title: 'Low Oxygen Saturation',
    description: 'SpO2 has dropped to 92%, immediate attention required',
    timestamp: '5 min ago',
    acknowledged: false,
    escalated: false
  },
  {
    id: '3',
    patientId: '2',
    patientName: 'Mary Davis',
    bedNumber: '102',
    wardLocation: 'General Ward',
    alertType: 'manual',
    priority: 'medium',
    title: 'Nurse Concern',
    description: 'Patient reports increased shortness of breath',
    timestamp: '15 min ago',
    acknowledged: true,
    assignedTo: 'Mike Wilson',
    escalated: false
  }
];

export default function Dashboard() {
  // Real-time data from WebSocket
  const { patients: livePatients } = usePatientUpdates();
  const { alerts: liveAlerts, acknowledgeAlert } = useAlertUpdates();
  const systemStatus = useSystemStatus();
  const { isConnected, connectionStatus, error } = useWebSocket();

  // Fallback to mock data if no live data available
  const [patients, setPatients] = useState<Patient[]>(mockPatients);
  const [alerts, setAlerts] = useState<Alert[]>(mockAlerts);

  // Memoize live data to prevent unnecessary updates
  const currentPatients = useMemo(() => {
    return livePatients.length > 0 ? livePatients : mockPatients;
  }, [livePatients]);

  const currentAlerts = useMemo(() => {
    return liveAlerts.length > 0 ? liveAlerts : mockAlerts;
  }, [liveAlerts]);

  // Update state only when actually needed
  useEffect(() => {
    setPatients(currentPatients);
  }, [currentPatients]);

  useEffect(() => {
    setAlerts(currentAlerts);
  }, [currentAlerts]);

  const handlePatientClick = (patient: Patient) => {
    window.location.href = `/patients/${patient.id}`;
  };

  const handleAlertClick = (alert: Alert) => {
    window.location.href = `/alerts/${alert.id}`;
  };

  const handleAcknowledgeAlert = (alertId: string) => {
    acknowledgeAlert(alertId);
  };

  const handleNavigation = (href: string) => {
    window.location.href = href;
  };

  const handleEmergency = () => {
    console.log('Emergency response activated');
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Navigation */}
      <SimpleNavbar
        userRole="charge_nurse"
        wardName="General Ward A"
      />

      {/* System Status Bar */}
      <div className="bg-white border-b px-6 py-2">
        <div className="flex items-center justify-between">
          <SystemStatus
            serverStatus={systemStatus.server}
            databaseStatus={systemStatus.database}
            websocketStatus={systemStatus.websocket}
            integrationStatus={systemStatus.integration}
            alertCount={alerts.length}
            activePatients={patients.length}
          />
          {/* Connection Status Indicator */}
          <div className="flex items-center gap-2 text-sm">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
            <span className={isConnected ? 'text-green-600' : 'text-red-600'}>
              {connectionStatus}
            </span>
            {error && (
              <Badge variant="destructive" className="text-xs">
                {error}
              </Badge>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-8">
        <div className="space-y-8">
          {/* Dashboard Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Ward Dashboard</h1>
              <p className="text-gray-600">Real-time patient monitoring and alert management</p>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500">
                {isConnected ? '🟢 Live data' : '🔴 Using cached data'}
              </div>
              <div className="text-xs text-gray-400">
                {isConnected ? 'Real-time updates active' : 'Attempting to reconnect...'}
              </div>
            </div>
          </div>

          {/* Alert Summary */}
          <AlertSummaryPanel
            alerts={alerts}
            onAlertClick={handleAlertClick}
            onAcknowledgeAlert={handleAcknowledgeAlert}
            onViewAllAlerts={() => handleNavigation('/alerts')}
          />

          {/* Patient Status Grid */}
          <div className="space-y-4">
            <h2 className="text-xl font-semibold text-gray-900">Patient Overview</h2>
            <PatientStatusGrid
              patients={patients}
              onPatientClick={handlePatientClick}
              onAddPatient={() => handleNavigation('/patients/add')}
            />
          </div>
        </div>
      </div>

      {/* Quick Actions Dock */}
      <QuickActionsDock
        userRole="charge_nurse"
        onAddPatient={() => handleNavigation('/patients/add')}
        onEnterVitals={() => console.log('Enter vitals')}
        onViewAlerts={() => handleNavigation('/alerts')}
        onEmergencyResponse={handleEmergency}
        onReports={() => handleNavigation('/analytics')}
        onTeams={() => handleNavigation('/users/teams')}
        onSettings={() => handleNavigation('/admin')}
        onSearch={() => console.log('Search')}
      />
    </div>
  );
}
