import React, { createContext, useCallback, useContext, useState } from 'react';
import { FaCheckCircle, FaInfoCircle, FaTimesCircle } from 'react-icons/fa';

export type ToastType = 'info' | 'success' | 'error';

interface Toast {
  id: number;
  type: ToastType;
  message: string;
}

interface ToastContextValue {
  push: (type: ToastType, message: string) => void;
}

const ToastContext = createContext<ToastContextValue>({
  // default no-op so calling without provider does not crash
  push: () => {},
});

export function useToast() {
  return useContext(ToastContext);
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const push = useCallback((type: ToastType, message: string) => {
    const id = Date.now();
    setToasts(ts => [...ts, { id, type, message }]);
    // auto-dismiss after 4 s
    setTimeout(() => {
      setToasts(ts => ts.filter(t => t.id !== id));
    }, 4000);
  }, []);

  return (
    <ToastContext.Provider value={{ push }}>
      {children}
      {/* Toast list */}
      <div className="fixed top-4 right-4 space-y-2 z-50">
        {toasts.map(t => (
          <div
            key={t.id}
            className={`flex items-center gap-2 px-4 py-2 rounded shadow text-sm text-white animate-slide-in
              ${t.type === 'error' ? 'bg-red-600' : t.type === 'success' ? 'bg-green-600' : 'bg-slate-800'}`}
          >
            {t.type === 'error' ? (
              <FaTimesCircle />
            ) : t.type === 'success' ? (
              <FaCheckCircle />
            ) : (
              <FaInfoCircle />
            )}
            <span>{t.message}</span>
          </div>
        ))}
      </div>
    </ToastContext.Provider>
  );
}

// Optional: simple slide-in animation
// Add this class inside global CSS (tailwind) if you want; otherwise remove the className above.
// .animate-slide-in { @apply transition-transform duration-300 ease-out translate-x-4 opacity-0; }
