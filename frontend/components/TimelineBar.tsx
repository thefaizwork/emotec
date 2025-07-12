import { getEmotionColor } from '../utils/colors';

interface Segment {
  emotion: string;
  start: number;
  end: number;
}
interface Props {
  segments: Segment[];
}
export default function TimelineBar({ segments }: Props) {
  if (segments.length === 0) return <p className="text-sm text-gray-400">No timeline data.</p>;
  const total = segments[segments.length - 1].end - segments[0].start || 1;
  return (
    <div className="w-full h-4 rounded bg-gray-700 overflow-hidden flex">
      {segments.map((s, i) => {
        const width = ((s.end - s.start) / total) * 100;
        return (
          <div
            key={i}
            title={`${s.emotion} (${s.start.toFixed(1)}s-${s.end.toFixed(1)}s)`}
            style={{ width: `${width}%`, backgroundColor: getEmotionColor(s.emotion) }}
          />
        );
      })}
    </div>
  );
}
