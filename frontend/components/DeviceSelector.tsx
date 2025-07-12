import { useState, useEffect } from 'react';

interface DeviceSelectorProps {
  kind: 'videoinput' | 'audioinput';
  onSelect: (deviceId: string) => void;
}

export default function DeviceSelector({ kind, onSelect }: DeviceSelectorProps) {
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);

  useEffect(() => {
    navigator.mediaDevices.enumerateDevices().then(list => {
      setDevices(list.filter(d => d.kind === kind));
    });
  }, [kind]);

  return (
    <select
      className="bg-gray-800 text-gray-100 p-2 rounded"
      onChange={e => onSelect(e.target.value)}
    >
      <option value="">Default {kind === 'videoinput' ? 'camera' : 'mic'}</option>
      {devices.map(d => (
        <option key={d.deviceId} value={d.deviceId}>{d.label || `${kind} ${d.deviceId}`}</option>
      ))}
    </select>
  );
}
