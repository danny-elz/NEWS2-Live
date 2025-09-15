'use client';

import { useState } from 'react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import {
  User,
  AlertTriangle,
  Activity,
  Heart,
  Thermometer,
  Wind,
  Eye,
  Clock,
  Bed
} from 'lucide-react';

export interface Patient {
  id: string;
  name: string;
  age: number;
  bedNumber: string;
  wardLocation: string;
  news2Score: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  vitals: {
    heartRate: number;
    bloodPressure: { systolic: number; diastolic: number };
    temperature: number;
    respiratoryRate: number;
    oxygenSaturation: number;
    avpu: 'A' | 'V' | 'P' | 'U';
  };
  lastUpdated: string;
  alerts: number;
  assignedNurse: string;
}

interface PatientStatusGridProps {
  patients: Patient[];
  onPatientClick?: (patient: Patient) => void;
  onAddPatient?: () => void;
  className?: string;
}

const getRiskBadgeVariant = (risk: string) => {
  switch (risk) {
    case 'critical':
      return 'destructive';
    case 'high':
      return 'destructive';
    case 'medium':
      return 'secondary';
    case 'low':
    default:
      return 'default';
  }
};

const getRiskColor = (risk: string) => {
  switch (risk) {
    case 'critical':
      return 'border-red-600 bg-red-50 hover:bg-red-100';
    case 'high':
      return 'border-red-400 bg-red-50 hover:bg-red-100';
    case 'medium':
      return 'border-amber-400 bg-amber-50 hover:bg-amber-100';
    case 'low':
    default:
      return 'border-green-400 bg-green-50 hover:bg-green-100';
  }
};

export function PatientStatusGrid({
  patients,
  onPatientClick,
  onAddPatient,
  className
}: PatientStatusGridProps) {
  const [selectedWard, setSelectedWard] = useState<string>('all');

  const wards = ['all', ...Array.from(new Set(patients.map(p => p.wardLocation)))];
  const filteredPatients = selectedWard === 'all'
    ? patients
    : patients.filter(p => p.wardLocation === selectedWard);

  return (
    <div className={cn('space-y-6', className)}>
      {/* Ward Filter */}
      <div className="flex flex-wrap gap-2">
        {wards.map((ward) => (
          <Button
            key={ward}
            variant={selectedWard === ward ? 'default' : 'outline'}
            size="sm"
            onClick={() => setSelectedWard(ward)}
            className="capitalize"
          >
            {ward === 'all' ? 'All Wards' : ward}
          </Button>
        ))}
      </div>

      {/* Patient Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {filteredPatients.map((patient) => (
          <div
            key={patient.id}
            className={cn(
              'relative p-4 rounded-lg border-2 cursor-pointer transition-all duration-200 transform hover:scale-105 hover:shadow-md',
              getRiskColor(patient.riskLevel)
            )}
            onClick={() => onPatientClick?.(patient)}
          >
            {/* Alert Badge */}
            {patient.alerts > 0 && (
              <div className="absolute -top-2 -right-2">
                <Badge variant="destructive" className="animate-pulse">
                  <AlertTriangle className="h-3 w-3 mr-1" />
                  {patient.alerts}
                </Badge>
              </div>
            )}

            {/* Patient Header */}
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="p-2 bg-white rounded-full">
                  <User className="h-4 w-4" />
                </div>
                <div>
                  <h3 className="font-semibold text-sm">{patient.name}</h3>
                  <p className="text-xs text-muted-foreground">Age {patient.age}</p>
                </div>
              </div>
              <Badge variant={getRiskBadgeVariant(patient.riskLevel)}>
                {patient.news2Score}
              </Badge>
            </div>

            {/* Bed and Ward Info */}
            <div className="flex items-center gap-4 mb-3 text-xs text-muted-foreground">
              <div className="flex items-center gap-1">
                <Bed className="h-3 w-3" />
                <span>Bed {patient.bedNumber}</span>
              </div>
              <div className="flex items-center gap-1">
                <span>{patient.wardLocation}</span>
              </div>
            </div>

            {/* Vital Signs Grid */}
            <div className="grid grid-cols-2 gap-2 mb-3">
              <div className="flex items-center gap-1 text-xs">
                <Heart className="h-3 w-3 text-red-500" />
                <span>{patient.vitals.heartRate} bpm</span>
              </div>
              <div className="flex items-center gap-1 text-xs">
                <Activity className="h-3 w-3 text-blue-500" />
                <span>{patient.vitals.bloodPressure.systolic}/{patient.vitals.bloodPressure.diastolic}</span>
              </div>
              <div className="flex items-center gap-1 text-xs">
                <Thermometer className="h-3 w-3 text-orange-500" />
                <span>{patient.vitals.temperature}°C</span>
              </div>
              <div className="flex items-center gap-1 text-xs">
                <Wind className="h-3 w-3 text-cyan-500" />
                <span>{patient.vitals.respiratoryRate}/min</span>
              </div>
              <div className="flex items-center gap-1 text-xs">
                <Eye className="h-3 w-3 text-purple-500" />
                <span>O2: {patient.vitals.oxygenSaturation}%</span>
              </div>
              <div className="flex items-center gap-1 text-xs">
                <span className="font-mono font-semibold">AVPU: {patient.vitals.avpu}</span>
              </div>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-between text-xs text-muted-foreground border-t pt-2">
              <div className="flex items-center gap-1">
                <Clock className="h-3 w-3" />
                <span>{patient.lastUpdated}</span>
              </div>
              <div className="text-right">
                <div>{patient.assignedNurse}</div>
              </div>
            </div>
          </div>
        ))}

        {/* Add Patient Card */}
        <div
          className="p-4 rounded-lg border-2 border-dashed border-gray-300 cursor-pointer transition-all duration-200 hover:border-gray-400 hover:bg-gray-50 flex flex-col items-center justify-center min-h-[200px]"
          onClick={onAddPatient}
        >
          <User className="h-8 w-8 text-gray-400 mb-2" />
          <span className="text-sm text-gray-500 font-medium">Add New Patient</span>
        </div>
      </div>

      {/* Summary */}
      <div className="flex items-center justify-between text-sm text-muted-foreground bg-gray-50 p-4 rounded-lg">
        <div>
          Total Patients: {filteredPatients.length}
        </div>
        <div className="flex gap-4">
          <span className="text-red-600">
            Critical: {filteredPatients.filter(p => p.riskLevel === 'critical').length}
          </span>
          <span className="text-amber-600">
            High Risk: {filteredPatients.filter(p => p.riskLevel === 'high').length}
          </span>
          <span className="text-blue-600">
            Active Alerts: {filteredPatients.reduce((sum, p) => sum + p.alerts, 0)}
          </span>
        </div>
      </div>
    </div>
  );
}