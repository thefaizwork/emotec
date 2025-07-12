import { useEffect, useState } from 'react';
import Head from 'next/head';
import Link from 'next/link';
import { listSessions, deleteSession } from '../lib/api';

type Session = {
  id: string;
  created_at: string;
  facial_summary: Record<string, number>;
  audio_summary: Record<string, number>;
};

export default function Analysis() {
  const [sessions, setSessions] = useState<Session[]>([]);

  const fetchSessions = async () => {
    const res = await listSessions();
    setSessions(res.data.sessions);
  };

  useEffect(() => {
    fetchSessions();
  }, []);

  const del = async (id: string) => {
    await deleteSession(id);
    fetchSessions();
  };

  return (
    <>
      <Head>
        <title>Analysis</title>
      </Head>
      <main className="p-6 max-w-4xl mx-auto flex flex-col gap-4">
        <h1 className="text-2xl font-bold">Session Analysis</h1>
        {sessions.length === 0 ? (
          <p>No sessions yet.</p>
        ) : (
          <table className="w-full text-left">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="p-2">Date</th>
                <th className="p-2">Facial</th>
                <th className="p-2">Audio</th>
                <th className="p-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map(s => (
                <tr key={s.id} className="border-b border-gray-800">
                  <td className="p-2">{new Date(s.created_at).toLocaleString()}</td>
                  <td className="p-2">{Object.entries(s.facial_summary).map(([k,v])=>`${k}:${v}`).join(', ')}</td>
                  <td className="p-2">{Object.entries(s.audio_summary).map(([k,v])=>`${k}:${v}`).join(', ')}</td>
                  <td className="p-2 flex gap-2">
                    <Link href={`/analysis/${s.id}`} className="underline">Detail</Link>
                    <button onClick={()=>del(s.id)} className="text-red-500">Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </main>
    </>
  );
}
