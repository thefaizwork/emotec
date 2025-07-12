import Link from 'next/link';
import { useRouter } from 'next/router';

export default function NavBar() {
  const { pathname } = useRouter();
  const linkCls = (path: string) =>
    `px-3 py-2 rounded hover:bg-gray-700 ${pathname === path ? 'bg-gray-800' : ''}`;

  return (
    <header className="bg-gray-900 text-gray-100 shadow-md">
      <nav className="max-w-4xl mx-auto flex gap-4 p-4">
        <Link href="/" className={linkCls('/')}>Home</Link>
        <Link href="/analysis" className={linkCls('/analysis')}>Analysis</Link>
        <Link href="/settings" className={linkCls('/settings')}>Settings</Link>
      </nav>
    </header>
  );
}
