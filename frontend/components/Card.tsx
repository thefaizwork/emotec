import React from 'react';
import { cva, VariantProps } from 'class-variance-authority';
import clsx from 'clsx';

const card = cva('rounded-lg border border-gray-800 bg-gradient-to-b from-gray-900 to-gray-950 shadow-sm', {
  variants: {
    padding: {
      sm: 'p-4',
      md: 'p-6',
      lg: 'p-8',
    },
  },
  defaultVariants: { padding: 'md' },
});

type CardProps = React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof card>;

export function Card({ className, padding, ...props }: CardProps) {
  return <div className={clsx(card({ padding }), className)} {...props} />;
}

export function CardTitle({ children }: { children: React.ReactNode }) {
  return <h2 className="text-lg font-semibold mb-2">{children}</h2>;
}

export function CardContent({ children }: { children: React.ReactNode }) {
  return <div className="text-sm text-gray-300 space-y-2">{children}</div>;
}
