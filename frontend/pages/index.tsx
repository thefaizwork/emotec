import { useState, useRef } from 'react';
import Head from 'next/head';
import DeviceSelector from '../components/DeviceSelector';
import AudioVisualizer from '../components/AudioVisualizer';
import axios from 'axios';
import AnimatedButton from '../components/AnimatedButton';
import { useToast } from '../components/Toast';

export default function Home() {
  const toast = useToast();
  const [videoId, setVideoId] = useState<string>('');
  const [audioId, setAudioId] = useState<string>('');
  const [running, setRunning] = useState(false);
  // timer & recorder refs so we can stop them later
  const frameTimer = useRef<ReturnType<typeof setInterval> | null>(null);
  const audioRecorder = useRef<MediaRecorder | null>(null);
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || 'https://emotec.onrender.com';
  const [stream, setStream] = useState<MediaStream | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const start = async () => {
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

      // -------- face frames --------
      const canvas = document.createElement('canvas');
      canvas.width = 224;
      canvas.height = 224;
      const sendFrame = async () => {
        if (!videoRef.current) return;
        const ctx = canvas.getContext('2d')!;
        ctx.drawImage(videoRef.current, 0, 0, 224, 224);
        canvas.toBlob(async blob => {
          if (!blob) return;
          try {
            const fd = new FormData();
            fd.append('image', blob, 'frame.jpg');
            await axios.post(`${API_BASE}/predict/face`, fd);
          } catch (err) {
            console.error('face req failed', err);
          }
        }, 'image/jpeg', 0.8);
      };
      frameTimer.current = setInterval(sendFrame, 500);

      // -------- audio chunks --------
      const recorder = new MediaRecorder(s, { mimeType: 'audio/webm' });
      audioRecorder.current = recorder;
      recorder.ondataavailable = async ev => {
        if (!ev.data.size) return;
        try {
          const fdA = new FormData();
          fdA.append('audio', ev.data, 'chunk.webm');
          await axios.post(`${API_BASE}/predict/audio`, fdA);
        } catch (err) {
          console.error('audio req failed', err);
        }
      };
      recorder.start(1000);

      setRunning(true);
    } catch (err) {
      toast.push('error', 'Failed to start session: ' + (err as any));
    }
  };

  const stop = () => {
    try {
      if (frameTimer.current !== null) clearInterval(frameTimer.current);
      if (audioRecorder.current && audioRecorder.current.state !== 'inactive') {
        audioRecorder.current.stop();
      }
      stream?.getTracks().forEach(t => t.stop());
      setStream(null);
      setRunning(false);
      toast.push('success', 'Stopped');
    } catch (err) {
      toast.push('error', 'Failed to stop: ' + (err as any));
    }
  };

  return (
    <>
      <Head>
        <title>Emotion Demo</title>
      </Head>
      <main className="flex flex-col gap-4 p-6 max-w-4xl mx-auto">
        <h1 className="text-2xl font-bold">Emotion Detection Demo</h1>
        <div className="flex flex-wrap gap-4 items-center">
          <DeviceSelector kind="videoinput" onSelect={setVideoId} />
          <DeviceSelector kind="audioinput" onSelect={setAudioId} />
          {!running ? (
             <AnimatedButton label="Start" onClick={start} variant="start" />
           ) : (
             <AnimatedButton label="Stop" onClick={stop} variant="stop" />
           )}
        </div>
        <video ref={videoRef} autoPlay muted playsInline className="w-full rounded bg-black" />
        <AudioVisualizer stream={stream} />
      </main>
    </>
  );
}
