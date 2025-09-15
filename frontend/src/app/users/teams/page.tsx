'use client';

import { SimpleNavbar } from '@/components/layout/simple-navbar';
import { QuickActionsDock } from '@/components/layout/quick-actions-dock';
import { SystemStatus } from '@/components/layout/system-status';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Users, UserPlus, Settings, Clock } from 'lucide-react';

const mockTeams = [
  {
    id: '1',
    name: 'ICU Team Alpha',
    ward: 'ICU',
    shift: 'Day Shift',
    members: [
      { name: 'Sarah Johnson', role: 'Charge Nurse', status: 'online' },
      { name: 'Dr. Michael Chen', role: 'Doctor', status: 'online' },
      { name: 'Emma Davis', role: 'Nurse', status: 'online' },
      { name: 'James Wilson', role: 'Nurse', status: 'break' },
    ]
  },
  {
    id: '2',
    name: 'General Ward Team',
    ward: 'General Ward',
    shift: 'Day Shift',
    members: [
      { name: 'Mike Wilson', role: 'Charge Nurse', status: 'online' },
      { name: 'Dr. Lisa Rodriguez', role: 'Doctor', status: 'offline' },
      { name: 'Anna Smith', role: 'Nurse', status: 'online' },
      { name: 'John Brown', role: 'Nurse', status: 'online' },
    ]
  },
  {
    id: '3',
    name: 'Cardiology Team',
    ward: 'Cardiology',
    shift: 'Day Shift',
    members: [
      { name: 'Lisa Chen', role: 'Charge Nurse', status: 'online' },
      { name: 'Dr. Robert Taylor', role: 'Doctor', status: 'online' },
      { name: 'Maria Garcia', role: 'Nurse', status: 'online' },
    ]
  }
];

const getStatusColor = (status: string) => {
  switch (status) {
    case 'online': return 'bg-green-500';
    case 'break': return 'bg-amber-500';
    case 'offline': default: return 'bg-gray-400';
  }
};

export default function TeamsPage() {
  return (
    <div className="min-h-screen bg-gray-50">
      <SimpleNavbar
        userRole="charge_nurse"
        wardName="Team Management"
      />

      <div className="bg-white border-b px-6 py-2">
        <SystemStatus alertCount={12} activePatients={45} />
      </div>

      <div className="container mx-auto px-6 py-8">
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Team Management</h1>
              <p className="text-gray-600">Manage ward teams and staff assignments</p>
            </div>
            <Button className="bg-blue-600 hover:bg-blue-700">
              <UserPlus className="h-4 w-4 mr-2" />
              Add Team Member
            </Button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            {mockTeams.map((team) => (
              <Card key={team.id} className="hover:shadow-md transition-shadow">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2">
                      <Users className="h-5 w-5" />
                      {team.name}
                    </CardTitle>
                    <Button variant="outline" size="sm">
                      <Settings className="h-4 w-4" />
                    </Button>
                  </div>
                  <div className="flex items-center gap-4 text-sm text-gray-600">
                    <span>{team.ward}</span>
                    <div className="flex items-center gap-1">
                      <Clock className="h-3 w-3" />
                      <span>{team.shift}</span>
                    </div>
                  </div>
                </CardHeader>

                <CardContent>
                  <div className="space-y-3">
                    <h4 className="font-medium text-sm text-gray-700">Team Members ({team.members.length})</h4>

                    {team.members.map((member, index) => (
                      <div key={index} className="flex items-center justify-between p-2 bg-gray-50 rounded-lg">
                        <div className="flex items-center gap-3">
                          <div className="relative">
                            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
                              <span className="text-xs font-medium text-blue-700">
                                {member.name.split(' ').map(n => n[0]).join('')}
                              </span>
                            </div>
                            <div className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border-2 border-white ${getStatusColor(member.status)}`} />
                          </div>

                          <div>
                            <div className="font-medium text-sm">{member.name}</div>
                            <div className="text-xs text-gray-600">{member.role}</div>
                          </div>
                        </div>

                        <Badge
                          variant={member.status === 'online' ? 'default' : 'secondary'}
                          className="text-xs capitalize"
                        >
                          {member.status}
                        </Badge>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          {/* Summary Stats */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Total Staff</p>
                    <p className="text-2xl font-bold">
                      {mockTeams.reduce((sum, team) => sum + team.members.length, 0)}
                    </p>
                  </div>
                  <Users className="h-8 w-8 text-blue-500" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Online Now</p>
                    <p className="text-2xl font-bold text-green-600">
                      {mockTeams.reduce((sum, team) =>
                        sum + team.members.filter(m => m.status === 'online').length, 0
                      )}
                    </p>
                  </div>
                  <div className="w-8 h-8 rounded-full bg-green-500" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">On Break</p>
                    <p className="text-2xl font-bold text-amber-600">
                      {mockTeams.reduce((sum, team) =>
                        sum + team.members.filter(m => m.status === 'break').length, 0
                      )}
                    </p>
                  </div>
                  <div className="w-8 h-8 rounded-full bg-amber-500" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-600">Teams</p>
                    <p className="text-2xl font-bold">{mockTeams.length}</p>
                  </div>
                  <div className="w-8 h-8 rounded-full bg-purple-500" />
                </div>
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