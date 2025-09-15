'use client';

import * as React from 'react';
import { Button } from '@/components/ui/button';
import { useEffect, useState, useRef, useMemo, useCallback } from 'react';
import {
  NavigationMenu,
  NavigationMenuItem,
  NavigationMenuLink,
  NavigationMenuList,
} from '@/components/ui/navigation-menu';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { cn } from '@/lib/utils';

// NEWS2 Live logo component
const NEWS2Logo = (props: React.SVGAttributes<SVGElement>) => {
  return (
    <svg width='1em' height='1em' viewBox='0 0 40 40' fill='currentColor' xmlns='http://www.w3.org/2000/svg' {...props}>
      <circle cx="20" cy="20" r="18" stroke="currentColor" strokeWidth="2" fill="none"/>
      <text x="20" y="25" textAnchor="middle" fontSize="14" fontWeight="bold" fill="currentColor">N2</text>
    </svg>
  );
};

// Hamburger icon component
const HamburgerIcon = ({ className, ...props }: React.SVGAttributes<SVGElement>) => (
  <svg
    className={cn('pointer-events-none', className)}
    width={16}
    height={16}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    xmlns="http://www.w3.org/2000/svg"
    {...props}
  >
    <path
      d="M4 12L20 12"
      className="origin-center -translate-y-[7px] transition-all duration-300 ease-[cubic-bezier(.5,.85,.25,1.1)] group-aria-expanded:translate-x-0 group-aria-expanded:translate-y-0 group-aria-expanded:rotate-[315deg]"
    />
    <path
      d="M4 12H20"
      className="origin-center transition-all duration-300 ease-[cubic-bezier(.5,.85,.25,1.8)] group-aria-expanded:rotate-45"
    />
    <path
      d="M4 12H20"
      className="origin-center translate-y-[7px] transition-all duration-300 ease-[cubic-bezier(.5,.85,.25,1.1)] group-aria-expanded:translate-y-0 group-aria-expanded:rotate-[135deg]"
    />
  </svg>
);

// Types
export interface NEWS2NavLink {
  href: string;
  label: string;
  active?: boolean;
  role?: string[];
}

export interface NEWS2NavbarProps extends React.HTMLAttributes<HTMLElement> {
  userRole?: string;
  wardName?: string;
  onNavigate?: (href: string) => void;
  onEmergency?: () => void;
}

// Role-based navigation links - memoized to prevent re-renders
const navigationLinksCache: { [key: string]: NEWS2NavLink[] } = {};

const getNavigationLinks = (role: string): NEWS2NavLink[] => {
  if (navigationLinksCache[role]) {
    return navigationLinksCache[role];
  }

  const baseLinks = [
    { href: '/', label: 'Dashboard', active: true, role: ['all'] },
    { href: '/patients', label: 'Patients', role: ['nurse', 'doctor', 'charge_nurse'] },
    { href: '/alerts', label: 'Alerts', role: ['all'] },
  ];

  const roleSpecificLinks: { [key: string]: NEWS2NavLink[] } = {
    nurse: [
      { href: '/analytics/ward', label: 'Ward Analytics', role: ['nurse'] },
    ],
    doctor: [
      { href: '/analytics/clinical', label: 'Clinical Analytics', role: ['doctor'] },
    ],
    charge_nurse: [
      { href: '/users/teams', label: 'Team Management', role: ['charge_nurse'] },
      { href: '/analytics', label: 'Analytics', role: ['charge_nurse'] },
    ],
    admin: [
      { href: '/admin', label: 'System Admin', role: ['admin'] },
      { href: '/users', label: 'User Management', role: ['admin'] },
      { href: '/integrations', label: 'Integrations', role: ['admin'] },
    ],
  };

  const result = [
    ...baseLinks.filter(link =>
      link.role.includes('all') || link.role.includes(role)
    ),
    ...(roleSpecificLinks[role] || [])
  ];

  navigationLinksCache[role] = result;
  return result;
};

export const NEWS2Navbar = React.forwardRef<HTMLElement, NEWS2NavbarProps>(
  (
    {
      className,
      userRole = 'nurse',
      wardName = 'General Ward',
      onNavigate,
      onEmergency,
      ...props
    },
    ref
  ) => {
    const [isMobile, setIsMobile] = useState(false);
    const containerRef = useRef<HTMLElement>(null);
    const navigationLinks = useMemo(() => getNavigationLinks(userRole), [userRole]);

    useEffect(() => {
      const checkWidth = () => {
        if (containerRef.current) {
          const width = containerRef.current.offsetWidth;
          setIsMobile(width < 768);
        }
      };

      checkWidth();

      const resizeObserver = new ResizeObserver(checkWidth);
      if (containerRef.current) {
        resizeObserver.observe(containerRef.current);
      }

      return () => {
        resizeObserver.disconnect();
      };
    }, []);

    const combinedRef = React.useCallback((node: HTMLElement | null) => {
      containerRef.current = node;
      if (typeof ref === 'function') {
        ref(node);
      } else if (ref) {
        ref.current = node;
      }
    }, [ref]);

    return (
      <header
        ref={combinedRef}
        className={cn(
          'sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-4 md:px-6',
          className
        )}
        {...props}
      >
        <div className="container mx-auto flex h-16 max-w-screen-2xl items-center justify-between gap-4">
          {/* Left side */}
          <div className="flex items-center gap-2">
            {/* Mobile menu trigger */}
            {isMobile && (
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    className="group h-9 w-9 hover:bg-accent hover:text-accent-foreground"
                    variant="ghost"
                    size="icon"
                  >
                    <HamburgerIcon />
                  </Button>
                </PopoverTrigger>
                <PopoverContent align="start" className="w-48 p-2">
                  <NavigationMenu className="max-w-none">
                    <NavigationMenuList className="flex-col items-start gap-1">
                      {navigationLinks.map((link, index) => (
                        <NavigationMenuItem key={index} className="w-full">
                          <a
                            href={link.href}
                            className={cn(
                              "flex w-full items-center rounded-md px-3 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground cursor-pointer no-underline",
                              link.active
                                ? "bg-accent text-accent-foreground"
                                : "text-foreground/80"
                            )}
                          >
                            {link.label}
                          </a>
                        </NavigationMenuItem>
                      ))}
                    </NavigationMenuList>
                  </NavigationMenu>
                </PopoverContent>
              </Popover>
            )}
            {/* Main nav */}
            <div className="flex items-center gap-6">
              <a
                href="/"
                className="flex items-center space-x-2 text-primary hover:text-primary/90 transition-colors cursor-pointer"
              >
                <div className="text-2xl">
                  <NEWS2Logo />
                </div>
                <div className="hidden sm:flex flex-col items-start">
                  <span className="font-bold text-xl">NEWS2 Live</span>
                  <span className="text-xs text-muted-foreground">{wardName}</span>
                </div>
              </a>
              {/* Navigation menu */}
              {!isMobile && (
                <NavigationMenu className="flex">
                  <NavigationMenuList className="gap-1">
                    {navigationLinks.map((link, index) => (
                      <NavigationMenuItem key={index}>
                        <a
                          href={link.href}
                          className={cn(
                            "group inline-flex h-9 w-max items-center justify-center rounded-md px-4 py-2 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 cursor-pointer no-underline",
                            link.active
                              ? "bg-accent text-accent-foreground"
                              : "text-foreground/80 hover:text-foreground"
                          )}
                        >
                          {link.label}
                        </a>
                      </NavigationMenuItem>
                    ))}
                  </NavigationMenuList>
                </NavigationMenu>
              )}
            </div>
          </div>
          {/* Right side */}
          <div className="flex items-center gap-3">
            <a href="/users/profile">
              <Button
                variant="ghost"
                size="sm"
                className="text-sm font-medium hover:bg-accent hover:text-accent-foreground"
              >
                Profile
              </Button>
            </a>
            <Button
              variant="destructive"
              size="sm"
              className="text-sm font-medium px-4 h-9 rounded-md shadow-sm"
              onClick={onEmergency}
            >
              Emergency
            </Button>
          </div>
        </div>
      </header>
    );
  }
);

NEWS2Navbar.displayName = 'NEWS2Navbar';

export { NEWS2Logo, HamburgerIcon };