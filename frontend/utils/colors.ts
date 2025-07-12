export const emotionColors: Record<string, string> = {
  Happy: '#10b981',
  Sad: '#6366f1',
  Angry: '#ef4444',
  Neutral: '#9ca3af',
  Calm: '#0ea5e9',
  Fear: '#f97316',
  Surprise: '#eab308',
  Disgust: '#7c3aed',
};

export function getEmotionColor(emotion: string) {
  return emotionColors[emotion] ?? '#6b7280';
}
