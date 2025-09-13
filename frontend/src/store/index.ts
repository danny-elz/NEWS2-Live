import { configureStore } from '@reduxjs/toolkit';
import { useDispatch, useSelector, TypedUseSelectorHook } from 'react-redux';

// Import slices
import authSlice from './slices/authSlice';
import alertsSlice from './slices/alertsSlice';
import patientsSlice from './slices/patientsSlice';
import dashboardSlice from './slices/dashboardSlice';
import suppressionSlice from './slices/suppressionSlice';
import configSlice from './slices/configSlice';

export const store = configureStore({
  reducer: {
    auth: authSlice,
    alerts: alertsSlice,
    patients: patientsSlice,
    dashboard: dashboardSlice,
    suppression: suppressionSlice,
    config: configSlice,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types
        ignoredActions: [
          'persist/PERSIST',
          'persist/REHYDRATE',
          'persist/REGISTER',
        ],
      },
    }),
  devTools: process.env.NODE_ENV !== 'production',
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Typed hooks
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;