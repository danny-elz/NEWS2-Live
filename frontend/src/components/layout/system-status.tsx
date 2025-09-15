import type { ComponentProps, HTMLAttributes } from 'react';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { Server, Wifi, Database, Activity, AlertTriangle, CheckCircle } from 'lucide-react';

export type StatusProps = ComponentProps<typeof Badge> & {
  status: 'online' | 'offline' | 'maintenance' | 'degraded' | 'critical';
};

export const Status = ({ className, status, ...props }: StatusProps) => (
  <Badge
    className={cn('flex items-center gap-2', 'group', status, className)}
    variant="secondary"
    {...props}
  />
);

export type StatusIndicatorProps = HTMLAttributes<HTMLSpanElement>;

export const StatusIndicator = ({
  className,
  ...props
}: StatusIndicatorProps) => (
  <span className="relative flex h-2 w-2" {...props}>
    <span
      className={cn(
        'absolute inline-flex h-full w-full animate-ping rounded-full opacity-75',
        'group-[.online]:bg-emerald-500',
        'group-[.offline]:bg-red-500',
        'group-[.maintenance]:bg-blue-500',
        'group-[.degraded]:bg-amber-500',
        'group-[.critical]:bg-red-600'
      )}
    />
    <span
      className={cn(
        'relative inline-flex h-2 w-2 rounded-full',
        'group-[.online]:bg-emerald-500',
        'group-[.offline]:bg-red-500',
        'group-[.maintenance]:bg-blue-500',
        'group-[.degraded]:bg-amber-500',
        'group-[.critical]:bg-red-600'
      )}
    />
  </span>
);

export type StatusLabelProps = HTMLAttributes<HTMLSpanElement>;

export const StatusLabel = ({
  className,
  children,
  ...props
}: StatusLabelProps) => (
  <span className={cn('text-muted-foreground', className)} {...props}>
    {children ?? (
      <>
        <span className="hidden group-[.online]:block">Online</span>
        <span className="hidden group-[.offline]:block">Offline</span>
        <span className="hidden group-[.maintenance]:block">Maintenance</span>
        <span className="hidden group-[.degraded]:block">Degraded</span>
        <span className="hidden group-[.critical]:block">Critical</span>
      </>
    )}
  </span>
);

// NEWS2 System Status Component
interface SystemStatusProps {
  serverStatus?: 'online' | 'offline' | 'maintenance' | 'degraded' | 'critical';
  databaseStatus?: 'online' | 'offline' | 'maintenance' | 'degraded' | 'critical';
  websocketStatus?: 'online' | 'offline' | 'maintenance' | 'degraded' | 'critical';
  integrationStatus?: 'online' | 'offline' | 'maintenance' | 'degraded' | 'critical';
  alertCount?: number;
  activePatients?: number;
  className?: string;
}

export function SystemStatus({
  serverStatus = 'online',
  databaseStatus = 'online',
  websocketStatus = 'online',
  integrationStatus = 'online',
  alertCount = 0,
  activePatients = 0,
  className
}: SystemStatusProps) {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'online':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'degraded':
        return <AlertTriangle className="h-4 w-4 text-amber-500" />;
      case 'offline':
      case 'critical':
        return <AlertTriangle className="h-4 w-4 text-red-500" />;
      case 'maintenance':
        return <Activity className="h-4 w-4 text-blue-500" />;
      default:
        return <CheckCircle className="h-4 w-4 text-green-500" />;
    }
  };

  const systemComponents = [
    { name: 'Server', status: serverStatus, icon: <Server className="h-4 w-4" /> },
    { name: 'Database', status: databaseStatus, icon: <Database className="h-4 w-4" /> },
    { name: 'WebSocket', status: websocketStatus, icon: <Wifi className="h-4 w-4" /> },
    { name: 'Integration', status: integrationStatus, icon: <Activity className="h-4 w-4" /> },
  ];

  return (
    <div className={cn('flex items-center gap-4 text-sm', className)}>
      {/* System Components Status */}
      <div className="flex items-center gap-2">
        {systemComponents.map((component) => (
          <Status key={component.name} status={component.status as 'online' | 'offline' | 'maintenance' | 'degraded' | 'critical'} className="px-2 py-1">
            <StatusIndicator />
            <span className="flex items-center gap-1">
              {component.icon}
              <span className="hidden sm:inline">{component.name}</span>
            </span>
            <StatusLabel />
          </Status>
        ))}
      </div>

      {/* Metrics */}
      <div className="flex items-center gap-4 text-xs text-muted-foreground">
        <div className="flex items-center gap-1">
          <AlertTriangle className="h-3 w-3" />
          <span>{alertCount} alerts</span>
        </div>
        <div className="flex items-center gap-1">
          <Activity className="h-3 w-3" />
          <span>{activePatients} patients</span>
        </div>
      </div>
    </div>
  );
}

export { Status, StatusIndicator, StatusLabel };