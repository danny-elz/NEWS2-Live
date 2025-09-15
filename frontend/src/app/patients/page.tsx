'use client';

import { useState } from 'react';
import { SimpleNavbar } from '@/components/layout/simple-navbar';
import { QuickActionsDock } from '@/components/layout/quick-actions-dock';
import { SystemStatus } from '@/components/layout/system-status';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { cn } from '@/lib/utils';
import {
  Search,
  Filter,
  UserPlus,
  MoreVertical,
  Eye,
  Edit,
  AlertTriangle,
  Activity,
  Heart,
  Thermometer,
  Wind,
  User,
  Bed,
  Clock
} from 'lucide-react';
import { Patient } from '@/components/dashboard/patient-status-grid';

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
  },
  {
    id: '4',
    name: 'Emma Wilson',
    age: 34,
    bedNumber: '203',
    wardLocation: 'Maternity',
    news2Score: 4,
    riskLevel: 'medium',
    vitals: {
      heartRate: 92,
      bloodPressure: { systolic: 130, diastolic: 85 },
      temperature: 37.0,
      respiratoryRate: 19,
      oxygenSaturation: 97,
      avpu: 'A'
    },
    lastUpdated: '15 min ago',
    alerts: 1,
    assignedNurse: 'Anna Rodriguez'
  }
];

export default function PatientsPage() {
  const [patients] = useState<Patient[]>(mockPatients);
  const [searchQuery, setSearchQuery] = useState('');
  const [wardFilter, setWardFilter] = useState('all');
  const [riskFilter, setRiskFilter] = useState('all');
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('table');

  const filteredPatients = patients.filter(patient => {
    const matchesSearch = patient.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         patient.bedNumber.includes(searchQuery) ||
                         patient.assignedNurse.toLowerCase().includes(searchQuery.toLowerCase());

    const matchesWard = wardFilter === 'all' || patient.wardLocation === wardFilter;
    const matchesRisk = riskFilter === 'all' || patient.riskLevel === riskFilter;

    return matchesSearch && matchesWard && matchesRisk;
  });

  const wards = [...new Set(patients.map(p => p.wardLocation))];

  const getRiskBadgeVariant = (risk: string) => {
    switch (risk) {
      case 'critical': return 'destructive';
      case 'high': return 'destructive';
      case 'medium': return 'secondary';
      case 'low': default: return 'default';
    }
  };

  const handlePatientClick = (patientId: string) => {
    window.location.href = `/patients/${patientId}`;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <SimpleNavbar
        userRole="charge_nurse"
        wardName="All Wards"
      />

      <div className="bg-white border-b px-6 py-2">
        <SystemStatus
          alertCount={filteredPatients.reduce((sum, p) => sum + p.alerts, 0)}
          activePatients={filteredPatients.length}
        />
      </div>

      <div className="container mx-auto px-6 py-8">
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Patient Management</h1>
              <p className="text-gray-600">Manage and monitor all patients</p>
            </div>
            <Button
              onClick={() => window.location.href = '/patients/add'}
              className="bg-blue-600 hover:bg-blue-700"
            >
              <UserPlus className="h-4 w-4 mr-2" />
              Add New Patient
            </Button>
          </div>

          {/* Filters */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Search & Filter</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col md:flex-row gap-4">
                <div className="flex-1">
                  <div className="relative">
                    <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                    <Input
                      placeholder="Search patients, bed numbers, or nurses..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>

                <Select value={wardFilter} onValueChange={setWardFilter}>
                  <SelectTrigger className="w-full md:w-48">
                    <SelectValue placeholder="All Wards" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Wards</SelectItem>
                    {wards.map(ward => (
                      <SelectItem key={ward} value={ward}>{ward}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <Select value={riskFilter} onValueChange={setRiskFilter}>
                  <SelectTrigger className="w-full md:w-48">
                    <SelectValue placeholder="All Risk Levels" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Risk Levels</SelectItem>
                    <SelectItem value="critical">Critical</SelectItem>
                    <SelectItem value="high">High</SelectItem>
                    <SelectItem value="medium">Medium</SelectItem>
                    <SelectItem value="low">Low</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </CardContent>
          </Card>

          {/* Results Summary */}
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-600">
              Showing {filteredPatients.length} of {patients.length} patients
            </div>
            <div className="flex gap-2">
              <Button
                variant={viewMode === 'table' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('table')}
              >
                Table
              </Button>
              <Button
                variant={viewMode === 'grid' ? 'default' : 'outline'}
                size="sm"
                onClick={() => setViewMode('grid')}
              >
                Grid
              </Button>
            </div>
          </div>

          {/* Patient Table */}
          {viewMode === 'table' && (
            <Card>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="border-b bg-gray-50">
                    <tr>
                      <th className="text-left p-4 font-medium text-gray-700">Patient</th>
                      <th className="text-left p-4 font-medium text-gray-700">Bed</th>
                      <th className="text-left p-4 font-medium text-gray-700">Ward</th>
                      <th className="text-left p-4 font-medium text-gray-700">NEWS2</th>
                      <th className="text-left p-4 font-medium text-gray-700">Vitals</th>
                      <th className="text-left p-4 font-medium text-gray-700">Nurse</th>
                      <th className="text-left p-4 font-medium text-gray-700">Last Update</th>
                      <th className="text-left p-4 font-medium text-gray-700">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredPatients.map((patient) => (
                      <tr
                        key={patient.id}
                        className="border-b hover:bg-gray-50 cursor-pointer"
                        onClick={() => handlePatientClick(patient.id)}
                      >
                        <td className="p-4">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 bg-gray-100 rounded-full flex items-center justify-center">
                              <User className="h-5 w-5 text-gray-600" />
                            </div>
                            <div>
                              <div className="font-medium text-gray-900">{patient.name}</div>
                              <div className="text-sm text-gray-500">Age {patient.age}</div>
                            </div>
                            {patient.alerts > 0 && (
                              <Badge variant="destructive" className="ml-2">
                                <AlertTriangle className="h-3 w-3 mr-1" />
                                {patient.alerts}
                              </Badge>
                            )}
                          </div>
                        </td>
                        <td className="p-4">
                          <div className="flex items-center gap-1">
                            <Bed className="h-4 w-4 text-gray-400" />
                            <span>{patient.bedNumber}</span>
                          </div>
                        </td>
                        <td className="p-4">
                          <span className="text-sm">{patient.wardLocation}</span>
                        </td>
                        <td className="p-4">
                          <Badge variant={getRiskBadgeVariant(patient.riskLevel)}>
                            {patient.news2Score}
                          </Badge>
                        </td>
                        <td className="p-4">
                          <div className="flex items-center gap-3 text-xs">
                            <div className="flex items-center gap-1">
                              <Heart className="h-3 w-3 text-red-500" />
                              <span>{patient.vitals.heartRate}</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Activity className="h-3 w-3 text-blue-500" />
                              <span>{patient.vitals.bloodPressure.systolic}/{patient.vitals.bloodPressure.diastolic}</span>
                            </div>
                            <div className="flex items-center gap-1">
                              <Thermometer className="h-3 w-3 text-orange-500" />
                              <span>{patient.vitals.temperature}°C</span>
                            </div>
                          </div>
                        </td>
                        <td className="p-4">
                          <span className="text-sm">{patient.assignedNurse}</span>
                        </td>
                        <td className="p-4">
                          <div className="flex items-center gap-1 text-sm text-gray-500">
                            <Clock className="h-3 w-3" />
                            <span>{patient.lastUpdated}</span>
                          </div>
                        </td>
                        <td className="p-4">
                          <div className="flex items-center gap-2">
                            <Button size="sm" variant="outline">
                              <Eye className="h-3 w-3" />
                            </Button>
                            <Button size="sm" variant="outline">
                              <Edit className="h-3 w-3" />
                            </Button>
                            <Button size="sm" variant="outline">
                              <MoreVertical className="h-3 w-3" />
                            </Button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
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