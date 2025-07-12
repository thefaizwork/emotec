import dynamic from 'next/dynamic';
// @ts-ignore - dynamic imported component type
const Bar: any = dynamic(() => import('react-chartjs-2').then(mod => mod.Bar as any), { ssr: false });
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend } from 'chart.js';
ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

interface Props {
  data: Record<string, number>;
  title: string;
}

export default function BarChart({ data, title }: Props) {
  const labels = Object.keys(data);
  const values = Object.values(data);
  return (
    <div className="bg-gray-900 p-4 rounded">
      <Bar
        options={{
          responsive: true,
          plugins: { legend: { display: false }, title: { display: true, text: title, color: '#e5e7eb' } },
          scales: {
            x: { grid: { color: '#1f2937' }, ticks: { color: '#9ca3af' } },
            y: { grid: { color: '#1f2937' }, ticks: { color: '#9ca3af' } },
          },
        }}
        data={{
          labels,
          datasets: [
            {
              label: title,
              data: values,
              borderColor: '#e5e7eb',
              backgroundColor: 'rgba(255,255,255,0.1)',
              tension: 0.4,
              pointRadius: 3,
              pointBackgroundColor: '#e5e7eb',
            },
          ],
        }}
      />
    </div>
  );
}
