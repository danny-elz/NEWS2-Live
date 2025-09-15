'use client';

import { useState } from 'react';
import { SimpleNavbar } from '@/components/layout/simple-navbar';
import { QuickActionsDock } from '@/components/layout/quick-actions-dock';
import { SystemStatus } from '@/components/layout/system-status';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';
import {
  AlertTriangle,
  Bell,
  CheckCircle2,
  Clock,
  User,
  Settings,
  History,
  XCircle,
  AlertCircle,
  TrendingUp
} from 'lucide-react';
import { Alert } from '@/components/dashboard/alert-summary-panel';

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
    description: 'NEWS2 score has reached 12 indicating critical deterioration. Immediate medical attention required.',
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
    description: 'SpO2 has dropped to 92%, oxygen therapy may be required',
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
    description: 'Patient reports increased shortness of breath and chest discomfort',
    timestamp: '15 min ago',
    acknowledged: true,
    assignedTo: 'Mike Wilson',
    escalated: false
  },
  {
    id: '4',
    patientId: '4',
    patientName: 'Emma Wilson',
    bedNumber: '203',
    wardLocation: 'Maternity',
    alertType: 'vitals',
    priority: 'high',
    title: 'Elevated Heart Rate',
    description: 'Heart rate consistently above 100 bpm for the last 30 minutes',
    timestamp: '8 min ago',
    acknowledged: false,
    escalated: false
  }
];

const getPriorityIcon = (priority: string) => {
  switch (priority) {
    case 'critical':
      return <XCircle className="h-5 w-5 text-red-600" />;
    case 'high':
      return <AlertTriangle className="h-5 w-5 text-red-500" />;
    case 'medium':
      return <AlertCircle className="h-5 w-5 text-amber-500" />;
    case 'low':
    default:
      return <Bell className="h-5 w-5 text-blue-500" />;
  }
};

const getPriorityColor = (priority: string) => {
  switch (priority) {
    case 'critical':
      return 'border-l-red-600 bg-red-50 hover:bg-red-100';
    case 'high':
      return 'border-l-red-400 bg-red-50 hover:bg-red-100';
    case 'medium':
      return 'border-l-amber-400 bg-amber-50 hover:bg-amber-100';
    case 'low':
    default:
      return 'border-l-blue-400 bg-blue-50 hover:bg-blue-100';
  }
};

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<Alert[]>(mockAlerts);
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null);

  const activeAlerts = alerts.filter(a => !a.acknowledged);
  const acknowledgedAlerts = alerts.filter(a => a.acknowledged);

  const handleAcknowledgeAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert =>
      alert.id === alertId
        ? { ...alert, acknowledged: true, assignedTo: 'Current User' }
        : alert
    ));
  };

  const handleEscalateAlert = (alertId: string) => {
    setAlerts(prev => prev.map(alert =>
      alert.id === alertId
        ? { ...alert, escalated: true }
        : alert
    ));
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <SimpleNavbar
        userRole="charge_nurse"
        wardName="All Wards"
      />

      <div className="bg-white border-b px-6 py-2">
        <SystemStatus
          alertCount={activeAlerts.length}
          activePatients={12}
        />
      </div>

      <div className="container mx-auto px-6 py-8">
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Alert Management</h1>
              <p className="text-gray-600">Monitor and manage patient alerts</p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={() => window.location.href = '/alerts/history'}>
                <History className="h-4 w-4 mr-2" />
                History
              </Button>
              <Button variant="outline" onClick={() => window.location.href = '/alerts/settings'}>
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </Button>
            </div>
          </div>

          {/* Alert Statistics */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Total Active</p>
                    <p className="text-2xl font-bold text-red-600">{activeAlerts.length}</p>
                  </div>
                  <AlertTriangle className="h-8 w-8 text-red-500" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Critical</p>
                    <p className="text-2xl font-bold text-red-800">
                      {alerts.filter(a => a.priority === 'critical').length}
                    </p>
                  </div>
                  <XCircle className="h-8 w-8 text-red-600" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Escalated</p>
                    <p className="text-2xl font-bold text-purple-600">
                      {alerts.filter(a => a.escalated).length}
                    </p>
                  </div>
                  <TrendingUp className="h-8 w-8 text-purple-500" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Resolved</p>
                    <p className="text-2xl font-bold text-green-600">{acknowledgedAlerts.length}</p>
                  </div>
                  <CheckCircle2 className="h-8 w-8 text-green-500" />
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Alert Tabs */}
          <Tabs defaultValue="active" className="space-y-4">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="active" className="flex items-center gap-2">
                <AlertTriangle className="h-4 w-4" />
                Active Alerts ({activeAlerts.length})
              </TabsTrigger>
              <TabsTrigger value="resolved" className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4" />
                Resolved ({acknowledgedAlerts.length})
              </TabsTrigger>
            </TabsList>

            <TabsContent value="active" className="space-y-4">
              {activeAlerts.map((alert) => (
                <Card
                  key={alert.id}
                  className={cn(
                    'border-l-4 cursor-pointer transition-all duration-200 hover:shadow-md',
                    getPriorityColor(alert.priority)
                  )}
                  onClick={() => setSelectedAlert(alert)}
                >
                  <CardContent className="p-6">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-3">
                          {getPriorityIcon(alert.priority)}
                          <h3 className="text-lg font-semibold text-gray-900">{alert.title}</h3>
                          <Badge
                            variant={alert.priority === 'critical' ? 'destructive' : 'secondary'}
                            className="capitalize"
                          >
                            {alert.priority}
                          </Badge>
                          {alert.escalated && (
                            <Badge variant="destructive">Escalated</Badge>
                          )}
                        </div>

                        <p className="text-gray-700 mb-4">{alert.description}</p>

                        <div className="flex items-center gap-6 text-sm text-gray-500">
                          <div className="flex items-center gap-1">
                            <User className="h-4 w-4" />
                            <span>{alert.patientName} - Bed {alert.bedNumber}</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Clock className="h-4 w-4" />
                            <span>{alert.timestamp}</span>
                          </div>
                          <div className="capitalize">
                            {alert.alertType} Alert
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-col gap-2 ml-4">
                        <Button
                          onClick={(e) => {
                            e.stopPropagation();
                            handleAcknowledgeAlert(alert.id);
                          }}
                          size="sm"
                          className="bg-green-600 hover:bg-green-700"
                        >
                          <CheckCircle2 className="h-4 w-4 mr-2" />
                          Acknowledge
                        </Button>
                        {!alert.escalated && (
                          <Button
                            onClick={(e) => {
                              e.stopPropagation();
                              handleEscalateAlert(alert.id);
                            }}
                            size="sm"
                            variant="outline"
                          >
                            <TrendingUp className="h-4 w-4 mr-2" />
                            Escalate
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </TabsContent>

            <TabsContent value="resolved" className="space-y-4">
              {acknowledgedAlerts.map((alert) => (
                <Card
                  key={alert.id}
                  className="border-l-4 border-l-green-400 bg-green-50 opacity-75"
                >
                  <CardContent className="p-6">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-3 mb-3">
                          <CheckCircle2 className="h-5 w-5 text-green-600" />
                          <h3 className="text-lg font-semibold text-gray-900">{alert.title}</h3>
                          <Badge variant="secondary">Resolved</Badge>
                        </div>

                        <p className="text-gray-700 mb-4">{alert.description}</p>

                        <div className="flex items-center gap-6 text-sm text-gray-500">
                          <div className="flex items-center gap-1">
                            <User className="h-4 w-4" />
                            <span>{alert.patientName} - Bed {alert.bedNumber}</span>
                          </div>
                          <div className="flex items-center gap-1">
                            <Clock className="h-4 w-4" />
                            <span>{alert.timestamp}</span>
                          </div>
                          {alert.assignedTo && (
                            <div>
                              Resolved by: {alert.assignedTo}
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </TabsContent>
          </Tabs>
        </div>
      </div>

      <QuickActionsDock
        userRole="charge_nurse"
        onAddPatient={() => window.location.href = '/patients/add'}
        onEnterVitals={() => console.log('Enter vitals')}
        onViewAlerts={() => window.location.href = '/alerts'}
        onEmergencyResponse={() => console.log('Emergency')}
        onReports={() => window.location.href = '/analytics'}
        onTeams={() => window.location.href = '/users/teams'}
        onSettings={() => window.location.href = '/admin'}
        onSearch={() => console.log('Search')}
      />
    </div>
  );
}