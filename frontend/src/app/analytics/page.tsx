'use client';

import { SimpleNavbar } from '@/components/layout/simple-navbar';
import { QuickActionsDock } from '@/components/layout/quick-actions-dock';
import { SystemStatus } from '@/components/layout/system-status';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';

const mockData = [
  { name: 'ICU', alerts: 12, patients: 15 },
  { name: 'General', alerts: 8, patients: 25 },
  { name: 'Cardiology', alerts: 6, patients: 18 },
  { name: 'Maternity', alerts: 3, patients: 12 },
];

const COLORS = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444'];

export default function AnalyticsPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <SimpleNavbar
        userRole="charge_nurse"
        wardName="Analytics Dashboard"
      />

      <div className="bg-white border-b px-6 py-2">
        <SystemStatus alertCount={29} activePatients={70} />
      </div>

      <div className="container mx-auto px-6 py-8">
        <div className="space-y-6">
          <div>
            <h1 className="text-3xl font-bold text-gray-900">Analytics Dashboard</h1>
            <p className="text-gray-600">Performance metrics and system insights</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Total Patients</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">70</div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Active Alerts</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-red-600">29</div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">Response Time</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">2.3m</div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-sm">System Uptime</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-600">99.8%</div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Alerts by Ward</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={mockData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="alerts" fill="#3B82F6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Patient Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={mockData}
                      dataKey="patients"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={80}
                      fill="#8884d8"
                    >
                      {mockData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
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