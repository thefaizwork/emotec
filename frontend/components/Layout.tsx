import Link from 'next/link';
import { useRouter } from 'next/router';
import { FaRegSmileBeam, FaChartBar, FaHome } from 'react-icons/fa';
import ThemeToggle from './ThemeToggle';
import clsx from 'clsx';
import React from 'react';

const navItems = [
  { label: 'Dashboard', href: '/', icon: <FaHome /> },
  { label: 'Analysis', href: '/analysis', icon: <FaChartBar /> },
];

export default function Layout({ children }: { children: React.ReactNode }) {
  const { pathname } = useRouter();

  return (
    <div className="min-h-screen bg-[#0d0d0d] text-gray-100 flex">
      {/* Sidebar */}
      <aside className="w-56 bg-[#111111] border-r border-gray-800 flex flex-col">
        <div className="px-4 py-5 flex items-center gap-2 text-xl font-semibold">
          <FaRegSmileBeam className="text-indigo-500" /> EmoTech
        </div>
        <nav className="flex-1 px-2 space-y-1">
          {navItems.map(item => (
            <Link
              key={item.href}
              href={item.href}
              className={clsx(
                'flex items-center gap-3 px-3 py-2 rounded-md hover:bg-gray-800 transition-colors',
                pathname === item.href ? 'bg-gray-800' : 'text-gray-400',
              )}
            >
              {item.icon}
              <span className="text-sm font-medium">{item.label}</span>
            </Link>
          ))}
        </nav>
        <p className="text-[10px] text-gray-500 px-4 pb-4">© 2025 EmoTech</p>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-x-hidden">
        {/* Top bar */}
        <header className="h-12 border-b border-gray-800 bg-[#111111] flex items-center px-6 shadow-sm sticky top-0 z-10 justify-between">
          <h1 className="text-lg font-semibold capitalize">
            {navItems.find(n => n.href === (pathname.startsWith('/analysis') ? '/analysis' : pathname))?.label ?? 'Page'}
          </h1>
          {/* Theme toggle */}
          <ThemeToggle />
        </header>
        <div className="p-6">{children}</div>
      </main>
    </div>
  );
}
