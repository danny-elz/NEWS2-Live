'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import {
  Eye,
  EyeOff,
  Lock,
  User,
  AlertTriangle,
  Shield,
  Activity,
  Heart
} from 'lucide-react';

// Get aurora background component
import { mcp__shadcn__getComponent } from '@/components/ui/aurora-background';

const NEWS2Logo = () => (
  <div className="flex items-center gap-3 mb-8">
    <div className="relative">
      <div className="w-12 h-12 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 flex items-center justify-center">
        <Heart className="h-6 w-6 text-white" />
      </div>
      <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full border-2 border-white">
        <Activity className="h-2 w-2 text-white m-0.5" />
      </div>
    </div>
    <div>
      <h1 className="text-2xl font-bold text-gray-900">NEWS2 Live</h1>
      <p className="text-sm text-gray-600">Patient Deterioration Detection</p>
    </div>
  </div>
);

export default function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [selectedRole, setSelectedRole] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  const roles = [
    { id: 'nurse', name: 'Ward Nurse', color: 'bg-blue-100 text-blue-700 border-blue-200' },
    { id: 'charge_nurse', name: 'Charge Nurse', color: 'bg-purple-100 text-purple-700 border-purple-200' },
    { id: 'doctor', name: 'Doctor', color: 'bg-green-100 text-green-700 border-green-200' },
    { id: 'admin', name: 'Administrator', color: 'bg-red-100 text-red-700 border-red-200' },
    { id: 'rapid_response', name: 'Rapid Response', color: 'bg-orange-100 text-orange-700 border-orange-200' }
  ];

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    // Simulate login
    setTimeout(() => {
      console.log('Login:', { email, role: selectedRole });
      setIsLoading(false);
      // Redirect to dashboard
      window.location.href = '/';
    }, 2000);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50 flex items-center justify-center p-4">
      {/* Background Pattern */}
      <div className="absolute inset-0 bg-grid-gray-900/[0.04] bg-[size:20px_20px]" />

      <div className="w-full max-w-md relative z-10">
        {/* Logo */}
        <div className="text-center mb-8">
          <NEWS2Logo />
        </div>

        {/* Login Card */}
        <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
          <CardHeader className="text-center pb-4">
            <CardTitle className="text-2xl font-semibold text-gray-900">
              Welcome Back
            </CardTitle>
            <CardDescription className="text-gray-600">
              Sign in to access your healthcare dashboard
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleLogin} className="space-y-6">
              {/* Role Selection */}
              <div className="space-y-3">
                <Label className="text-sm font-medium text-gray-700">
                  Select Your Role
                </Label>
                <div className="grid grid-cols-1 gap-2">
                  {roles.map((role) => (
                    <button
                      key={role.id}
                      type="button"
                      onClick={() => setSelectedRole(role.id)}
                      className={cn(
                        'flex items-center justify-center p-3 rounded-lg border-2 transition-all duration-200 hover:shadow-md',
                        selectedRole === role.id
                          ? role.color + ' shadow-md scale-105'
                          : 'bg-gray-50 text-gray-700 border-gray-200 hover:bg-gray-100'
                      )}
                    >
                      <Shield className="h-4 w-4 mr-2" />
                      <span className="font-medium">{role.name}</span>
                    </button>
                  ))}
                </div>
              </div>

              {/* Email Input */}
              <div className="space-y-2">
                <Label htmlFor="email" className="text-sm font-medium text-gray-700">
                  Email Address
                </Label>
                <div className="relative">
                  <User className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                  <Input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="Enter your email"
                    className="pl-10 h-12"
                    required
                  />
                </div>
              </div>

              {/* Password Input */}
              <div className="space-y-2">
                <Label htmlFor="password" className="text-sm font-medium text-gray-700">
                  Password
                </Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                  <Input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    placeholder="Enter your password"
                    className="pl-10 pr-10 h-12"
                    required
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-3 text-gray-400 hover:text-gray-600"
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>

              {/* Login Button */}
              <Button
                type="submit"
                disabled={!selectedRole || !email || !password || isLoading}
                className="w-full h-12 text-base font-medium bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700 transition-all duration-200"
              >
                {isLoading ? (
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Signing In...
                  </div>
                ) : (
                  'Sign In'
                )}
              </Button>

              {/* Links */}
              <div className="text-center space-y-2">
                <button
                  type="button"
                  onClick={() => window.location.href = '/forgot-password'}
                  className="text-sm text-blue-600 hover:text-blue-700 hover:underline"
                >
                  Forgot your password?
                </button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Security Notice */}
        <div className="mt-6 p-4 bg-amber-50 border border-amber-200 rounded-lg">
          <div className="flex items-center gap-2 text-amber-800">
            <AlertTriangle className="h-4 w-4" />
            <span className="text-sm font-medium">Security Notice</span>
          </div>
          <p className="text-xs text-amber-700 mt-1">
            This system contains protected health information. Unauthorized access is prohibited.
          </p>
        </div>

        {/* Demo Credentials */}
        <div className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="text-sm font-medium text-blue-800 mb-2">Demo Credentials:</h4>
          <div className="text-xs text-blue-700 space-y-1">
            <div>Email: nurse@news2.demo</div>
            <div>Password: demo123</div>
            <div>Select any role above</div>
          </div>
        </div>
      </div>
    </div>
  );
}