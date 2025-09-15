'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { cn } from '@/lib/utils';
import {
  Mail,
  ArrowLeft,
  CheckCircle,
  Heart,
  Activity
} from 'lucide-react';

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

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    // Simulate API call
    setTimeout(() => {
      setIsLoading(false);
      setIsSubmitted(true);
    }, 2000);
  };

  if (isSubmitted) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50 flex items-center justify-center p-4">
        <div className="absolute inset-0 bg-grid-gray-900/[0.04] bg-[size:20px_20px]" />

        <div className="w-full max-w-md relative z-10">
          <div className="text-center mb-8">
            <NEWS2Logo />
          </div>

          <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
            <CardContent className="pt-8 pb-8">
              <div className="text-center space-y-4">
                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                  <CheckCircle className="h-8 w-8 text-green-600" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-gray-900 mb-2">
                    Check Your Email
                  </h2>
                  <p className="text-gray-600 text-sm">
                    We&apos;ve sent a password reset link to <strong>{email}</strong>
                  </p>
                </div>
                <div className="pt-4">
                  <Button
                    onClick={() => window.location.href = '/login'}
                    className="w-full"
                  >
                    <ArrowLeft className="h-4 w-4 mr-2" />
                    Back to Sign In
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-cyan-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-grid-gray-900/[0.04] bg-[size:20px_20px]" />

      <div className="w-full max-w-md relative z-10">
        <div className="text-center mb-8">
          <NEWS2Logo />
        </div>

        <Card className="shadow-lg border-0 bg-white/80 backdrop-blur-sm">
          <CardHeader className="text-center pb-4">
            <CardTitle className="text-2xl font-semibold text-gray-900">
              Reset Password
            </CardTitle>
            <CardDescription className="text-gray-600">
              Enter your email address and we&apos;ll send you a link to reset your password
            </CardDescription>
          </CardHeader>

          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="email" className="text-sm font-medium text-gray-700">
                  Email Address
                </Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
                  <Input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="Enter your email address"
                    className="pl-10 h-12"
                    required
                  />
                </div>
              </div>

              <Button
                type="submit"
                disabled={!email || isLoading}
                className="w-full h-12 text-base font-medium bg-gradient-to-r from-blue-600 to-cyan-600 hover:from-blue-700 hover:to-cyan-700"
              >
                {isLoading ? (
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Sending Reset Link...
                  </div>
                ) : (
                  'Send Reset Link'
                )}
              </Button>

              <div className="text-center">
                <button
                  type="button"
                  onClick={() => window.location.href = '/login'}
                  className="text-sm text-gray-600 hover:text-gray-800 hover:underline flex items-center justify-center gap-2"
                >
                  <ArrowLeft className="h-3 w-3" />
                  Back to Sign In
                </button>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}