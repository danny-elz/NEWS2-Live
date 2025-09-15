'use client';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import {
  AlertTriangle,
  Bell,
  Clock,
  User,
  TrendingUp,
  CheckCircle2,
  AlertCircle,
  XCircle
} from 'lucide-react';

export interface Alert {
  id: string;
  patientId: string;
  patientName: string;
  bedNumber: string;
  wardLocation: string;
  alertType: 'news2' | 'vitals' | 'manual' | 'system';
  priority: 'critical' | 'high' | 'medium' | 'low';
  title: string;
  description: string;
  timestamp: string;
  acknowledged: boolean;
  assignedTo?: string;
  responseTime?: string;
  escalated: boolean;
}

interface AlertSummaryPanelProps {
  alerts: Alert[];
  onAlertClick?: (alert: Alert) => void;
  onAcknowledgeAlert?: (alertId: string) => void;
  onViewAllAlerts?: () => void;
  className?: string;
}

const getPriorityIcon = (priority: string) => {
  switch (priority) {
    case 'critical':
      return <XCircle className="h-4 w-4 text-red-600" />;
    case 'high':
      return <AlertTriangle className="h-4 w-4 text-red-500" />;
    case 'medium':
      return <AlertCircle className="h-4 w-4 text-amber-500" />;
    case 'low':
    default:
      return <Bell className="h-4 w-4 text-blue-500" />;
  }
};

const getPriorityColor = (priority: string) => {
  switch (priority) {
    case 'critical':
      return 'border-l-red-600 bg-red-50';
    case 'high':
      return 'border-l-red-400 bg-red-50';
    case 'medium':
      return 'border-l-amber-400 bg-amber-50';
    case 'low':
    default:
      return 'border-l-blue-400 bg-blue-50';
  }
};

const getAlertTypeIcon = (type: string) => {
  switch (type) {
    case 'news2':
      return <TrendingUp className="h-3 w-3" />;
    case 'vitals':
      return <Activity className="h-3 w-3" />;
    case 'manual':
      return <User className="h-3 w-3" />;
    case 'system':
    default:
      return <Bell className="h-3 w-3" />;
  }
};

function Activity({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      height="24"
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="2"
      viewBox="0 0 24 24"
      width="24"
      xmlns="http://www.w3.org/2000/svg"
    >
      <path d="m22 12-4-4-4 4-6-6-6 6" />
      <path d="m16 8 4-4 4 4" />
    </svg>
  );
}

export function AlertSummaryPanel({
  alerts,
  onAlertClick,
  onAcknowledgeAlert,
  onViewAllAlerts,
  className
}: AlertSummaryPanelProps) {
  const sortedAlerts = [...alerts].sort((a, b) => {
    const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
    return priorityOrder[b.priority] - priorityOrder[a.priority];
  });

  const recentAlerts = sortedAlerts.slice(0, 5);

  const alertCounts = {
    total: alerts.length,
    critical: alerts.filter(a => a.priority === 'critical').length,
    high: alerts.filter(a => a.priority === 'high').length,
    acknowledged: alerts.filter(a => a.acknowledged).length,
    escalated: alerts.filter(a => a.escalated).length,
  };

  return (
    <div className={cn('bg-white rounded-lg border shadow-sm', className)}>
      {/* Header */}
      <div className="p-4 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-amber-500" />
            <h2 className="text-lg font-semibold">Alert Summary</h2>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={onViewAllAlerts}
          >
            View All
          </Button>
        </div>
      </div>

      {/* Alert Statistics */}
      <div className="p-4 border-b">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{alertCounts.total}</div>
            <div className="text-xs text-gray-500">Total Alerts</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">{alertCounts.critical}</div>
            <div className="text-xs text-gray-500">Critical</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-amber-600">{alertCounts.high}</div>
            <div className="text-xs text-gray-500">High Priority</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{alertCounts.acknowledged}</div>
            <div className="text-xs text-gray-500">Acknowledged</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">{alertCounts.escalated}</div>
            <div className="text-xs text-gray-500">Escalated</div>
          </div>
        </div>
      </div>

      {/* Recent Alerts List */}
      <div className="p-4">
        <h3 className="text-sm font-medium text-gray-700 mb-3">Recent Alerts</h3>
        <div className="space-y-2">
          {recentAlerts.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <CheckCircle2 className="h-8 w-8 mx-auto mb-2 text-green-500" />
              <p className="text-sm">No active alerts</p>
            </div>
          ) : (
            recentAlerts.map((alert) => (
              <div
                key={alert.id}
                className={cn(
                  'p-3 rounded-lg border-l-4 cursor-pointer transition-all duration-200 hover:shadow-md',
                  getPriorityColor(alert.priority),
                  alert.acknowledged ? 'opacity-60' : ''
                )}
                onClick={() => onAlertClick?.(alert)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      {getPriorityIcon(alert.priority)}
                      <span className="font-medium text-sm">{alert.title}</span>
                      <Badge variant="outline" className="text-xs">
                        {getAlertTypeIcon(alert.alertType)}
                        <span className="ml-1 capitalize">{alert.alertType}</span>
                      </Badge>
                      {alert.escalated && (
                        <Badge variant="destructive" className="text-xs">
                          Escalated
                        </Badge>
                      )}
                    </div>

                    <div className="text-xs text-gray-600 mb-2">
                      {alert.description}
                    </div>

                    <div className="flex items-center gap-4 text-xs text-gray-500">
                      <div className="flex items-center gap-1">
                        <User className="h-3 w-3" />
                        <span>{alert.patientName} - Bed {alert.bedNumber}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        <span>{alert.timestamp}</span>
                      </div>
                    </div>

                    {alert.assignedTo && (
                      <div className="text-xs text-gray-500 mt-1">
                        Assigned to: {alert.assignedTo}
                      </div>
                    )}
                  </div>

                  <div className="flex flex-col gap-2">
                    {!alert.acknowledged && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="text-xs h-6"
                        onClick={(e) => {
                          e.stopPropagation();
                          onAcknowledgeAlert?.(alert.id);
                        }}
                      >
                        Ack
                      </Button>
                    )}
                    {alert.acknowledged && (
                      <Badge variant="secondary" className="text-xs">
                        <CheckCircle2 className="h-3 w-3 mr-1" />
                        Ack&apos;d
                      </Badge>
                    )}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>

        {alerts.length > 5 && (
          <div className="text-center mt-4">
            <Button
              variant="ghost"
              size="sm"
              onClick={onViewAllAlerts}
              className="text-xs"
            >
              View {alerts.length - 5} more alerts
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}