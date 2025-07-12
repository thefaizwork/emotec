import { useState, useRef } from 'react';
import Head from 'next/head';
import DeviceSelector from '../components/DeviceSelector';
import AudioVisualizer from '../components/AudioVisualizer';
import { startSession, stopSession, healthCheck } from '../lib/api';
import { useToast } from '../components/Toast';

export default function Home() {
  const toast = useToast();
  const [videoId, setVideoId] = useState<string>('');
  const [audioId, setAudioId] = useState<string>('');
  const [running, setRunning] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const start = async () => {
    // Ensure backend API is reachable before attempting media capture
    try {
      await healthCheck();
    } catch (err) {
      toast.push('error', 'Cannot connect to backend API. Please make sure the server is running.');
      return;
    }

    try {
      const constraints: MediaStreamConstraints = {
        video: videoId ? { deviceId: { exact: videoId } } : true,
        audio: audioId ? { deviceId: { exact: audioId } } : true,
      };
      const s = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(s);
      if (videoRef.current) {
        videoRef.current.srcObject = s;
      }

      try {
        await startSession();
        setRunning(true);
      } catch (e: any) {
        if (e?.response?.status === 400) {
          // backend says already running – try to stop then start once
          try {
            await stopSession();
            await startSession();
            setRunning(true);
          } catch (inner) {
            toast.push('error', 'Backend is already running a session and could not be reset.');
          }
        } else {
          throw e;
        }
      }
    } catch (err) {
      toast.push('error', 'Failed to start session: ' + (err as any));
    }
  };

  const stop = async () => {
    try {
      stream?.getTracks().forEach(t => t.stop());
      setStream(null);
      const res = await stopSession();
      console.log(res.data);
      setRunning(false);
      toast.push('success', 'Session finished!');
    } catch (err) {
      toast.push('error', 'Failed to stop session: ' + (err as any));
    }
  };

  return (
    <>
      <Head>
        <title>Emotion Demo</title>
      </Head>
      <main className="flex flex-col gap-4 p-6 max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold">Emotion Detection Demo</h1>
        <div className="flex gap-4 items-center">
          <DeviceSelector kind="videoinput" onSelect={setVideoId} />
          <DeviceSelector kind="audioinput" onSelect={setAudioId} />
          {!running ? (
            <button onClick={start} className="px-4 py-2 bg-green-600 rounded">Start</button>
          ) : (
            <button onClick={stop} className="px-4 py-2 bg-red-600 rounded">Stop</button>
          )}
        </div>
        <video ref={videoRef} autoPlay muted playsInline className="w-full rounded bg-black" />
        <AudioVisualizer stream={stream} />
      </main>
    </>
  );
}
