import { useEffect, useRef } from 'react';

interface Props {
  stream?: MediaStream | null;
}

export default function AudioVisualizer({ stream }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!stream) return;

    const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(stream);
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    source.connect(analyser);

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext('2d')!;

    const draw = () => {
      requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      // Clear
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Create rainbow gradient
      const grad = ctx.createLinearGradient(0, 0, canvas.width, 0);
      grad.addColorStop(0, '#10b981'); // green
      grad.addColorStop(0.25, '#34d399');
      grad.addColorStop(0.5, '#8b5cf6');
      grad.addColorStop(0.75, '#ec4899');
      grad.addColorStop(1, '#f59e0b'); // yellow
      ctx.strokeStyle = grad;
      ctx.lineWidth = 2;

      ctx.beginPath();
      const sliceWidth = (canvas.width * 1.0) / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0; // [0,255] -> ~[0,2]
        const y = (v * canvas.height) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    };
    draw();

    return () => {
      audioCtx.close();
    };
  }, [stream]);

  return <canvas ref={canvasRef} className="w-full h-32 bg-gray-900 rounded" />;
}
