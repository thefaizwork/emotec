import React, { FC } from 'react';

interface AnimatedButtonProps {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  variant?: 'start' | 'stop';
}

// Tailwind gradient colours for start/stop variants
const variantStyles = {
  start: {
    gradient: 'from-green-400 via-green-500 to-green-600',
    ring: 'focus:ring-green-500',
  },
  stop: {
    gradient: 'from-red-400 via-red-500 to-red-600',
    ring: 'focus:ring-red-500',
  },
} as const;

const AnimatedButton: FC<AnimatedButtonProps> = ({
  label,
  onClick,
  disabled = false,
  variant = 'start',
}) => {
  const styles = variantStyles[variant];

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`relative inline-flex items-center justify-center p-0.5 overflow-hidden font-medium rounded-lg group ${styles.ring} focus:outline-none transition-transform active:scale-95 ${
        disabled ? 'opacity-50 cursor-not-allowed' : 'hover:scale-105'
      }`}
    >
      {/* animated gradient border */}
      <span
        className={`absolute inset-0 w-full h-full bg-gradient-to-br ${styles.gradient} group-hover:animate-pulse pointer-events-none`}
      />
      {/* inner label */}
      <span className="relative px-6 py-2.5 bg-gray-900 rounded-md text-white">
        {label}
      </span>
    </button>
  );
};

export default AnimatedButton;
