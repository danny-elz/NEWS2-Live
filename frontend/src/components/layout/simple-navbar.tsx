'use client';

import { Heart, Activity } from 'lucide-react';
import { Button } from '@/components/ui/button';

const NEWS2Logo = () => (
  <div className="relative">
    <div className="w-8 h-8 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 flex items-center justify-center">
      <Heart className="h-4 w-4 text-white" />
    </div>
    <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full border-2 border-white">
      <Activity className="h-1 w-1 text-white m-0.5" />
    </div>
  </div>
);

interface SimpleNavbarProps {
  userRole?: string;
  wardName?: string;
}

export function SimpleNavbar({ userRole = 'nurse', wardName = 'General Ward' }: SimpleNavbarProps) {
  const navLinks = [
    { href: '/', label: 'Dashboard' },
    { href: '/patients', label: 'Patients' },
    { href: '/alerts', label: 'Alerts' },
    { href: '/analytics', label: 'Analytics' },
    { href: '/users/teams', label: 'Teams' },
  ];

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-4 md:px-6">
      <div className="container mx-auto flex h-16 max-w-screen-2xl items-center justify-between gap-4">
        {/* Left side */}
        <div className="flex items-center gap-6">
          <a href="/" className="flex items-center space-x-3 text-primary hover:text-primary/90 transition-colors">
            <NEWS2Logo />
            <div>
              <span className="font-bold text-xl">NEWS2 Live</span>
              <div className="text-xs text-muted-foreground">{wardName}</div>
            </div>
          </a>

          {/* Navigation links */}
          <nav className="hidden md:flex items-center space-x-6">
            {navLinks.map((link) => (
              <a
                key={link.href}
                href={link.href}
                className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors"
              >
                {link.label}
              </a>
            ))}
          </nav>
        </div>

        {/* Right side */}
        <div className="flex items-center gap-3">
          <a href="/users/profile">
            <Button variant="ghost" size="sm">
              Profile
            </Button>
          </a>
          <Button variant="destructive" size="sm">
            Emergency
          </Button>
        </div>
      </div>
    </header>
  );
}