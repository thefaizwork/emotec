import { useRouter } from 'next/router';
import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { getSession } from '../../lib/api';



import { Card, CardTitle, CardContent } from '../../components/Card';
import BarChart from '../../components/BarChart';
import TimelineBar from '../../components/TimelineBar';
import { getEmotionColor } from '../../utils/colors';
import { FaRegLightbulb } from 'react-icons/fa';




export default function AnalysisDetail() {
  const router = useRouter();
  const { id } = router.query as { id?: string };
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    if (id) {
      getSession(id).then(res => setData(res.data));
    }
  }, [id]);

  if (!data) {
    return <p className="p-6">Loading...</p>;
  }

  const { facial_summary, audio_summary, facial_timeline, suggestions, created_at } = data;
  const topFacial = Object.entries(facial_summary).reduce((a:any,b:any)=> b[1] > (a?a[1]:0)? b : a, undefined as any)||['N/A',0];
  const topAudio = Object.entries(audio_summary).reduce((a:any,b:any)=> b[1] > (a?a[1]:0)? b : a, undefined as any)||['N/A',0];

  return (
    <>
      <Head><title>Session Analysis</title></Head>
      <main className="p-6 max-w-5xl mx-auto space-y-6">
        <h1 className="text-2xl font-bold">Session – {new Date(created_at).toLocaleString()}</h1>

        <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-4">
          <Card>
            <CardTitle>Top Facial Emotion</CardTitle>
            <CardContent>
              <p className="text-4xl font-bold" style={{color:getEmotionColor(topFacial[0])}}>
                {topFacial[0]}
              </p>
              <p className="text-sm text-gray-400">{topFacial[1]} occurrences</p>
            </CardContent>
          </Card>
          <Card>
            <CardTitle>Top Audio Emotion</CardTitle>
            <CardContent>
              <p className="text-4xl font-bold" style={{color:getEmotionColor(topAudio[0])}}>
                {topAudio[0]}
              </p>
              <p className="text-sm text-gray-400">{topAudio[1]} occurrences</p>
            </CardContent>
          </Card>
          <Card>
            <CardTitle>Suggestions <FaRegLightbulb className="inline ml-2"/></CardTitle>
            <CardContent>
              <ul className="list-disc pl-5 space-y-1">
                {suggestions.map((s:string,i:number)=><li key={i}>{s}</li>)}
              </ul>
            </CardContent>
          </Card>
        </div>

        <div className="grid md:grid-cols-2 gap-4">
          <BarChart data={facial_summary} title="Facial Emotions" />
          <BarChart data={audio_summary} title="Audio Emotions" />
        </div>

        <Card>
          <CardTitle>Facial Timeline</CardTitle>
          <CardContent>
            <TimelineBar segments={facial_timeline} />
          </CardContent>
        </Card>
      </main>
    </>
  );
}
  
