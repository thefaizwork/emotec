import Head from 'next/head';
import { useTheme } from 'next-themes';

export default function Settings() {
  const { theme, setTheme } = useTheme();
  return (
    <>
      <Head>
        <title>Settings</title>
      </Head>
      <main className="p-6 max-w-xl mx-auto flex flex-col gap-4">
        <h1 className="text-2xl font-bold">Settings</h1>
        <div className="flex gap-2 items-center">
          <span>Theme:</span>
          <select
            value={theme}
            onChange={e => setTheme(e.target.value)}
            className="bg-gray-800 text-gray-100 p-2 rounded"
          >
            <option value="system">System</option>
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </div>
      </main>
    </>
  );
}
